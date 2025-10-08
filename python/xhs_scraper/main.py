"""Command-line scraper for Xiaohongshu search notes.

This script fetches note data from the Xiaohongshu search result pages,
filters out video entries as well as notes below a configurable like-count
threshold, and finally exports the result to a CSV file.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests


USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.3 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_3 like Mac OS X) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.3 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0",
]

INITIAL_STATE_PATTERN = re.compile(
    r"window\.__INITIAL_(?:STATE|DATA)__\s*=\s*(\{.*?\})\s*;\s*</script>",
    re.DOTALL,
)


@dataclass
class Note:
    keyword: str
    note_id: str
    title: str = ""
    desc: str = ""
    author_name: str = ""
    author_id: str = ""
    likes: int = 0
    comments: int = 0
    release_time: str = ""
    url: str = ""
    cover_img: str = ""
    images: List[str] = field(default_factory=list)

    def to_row(self) -> Dict[str, object]:
        return {
            "keyword": self.keyword,
            "note_id": self.note_id,
            "title": self.title,
            "desc": self.desc,
            "author_name": self.author_name,
            "author_id": self.author_id,
            "likes": self.likes,
            "comments": self.comments,
            "release_time": self.release_time,
            "url": self.url,
            "cover_img": self.cover_img,
            "images": "|".join(self.images),
        }


class XiaohongshuScraper:
    """Scrape Xiaohongshu search results for note metadata."""

    BASE_URL = "https://www.xiaohongshu.com"
    SEARCH_URL = "https://www.xiaohongshu.com/search_result"

    def __init__(
        self,
        keyword: str,
        max_pages: int = 5,
        sort: str = "popularity",
        likes_min: int = 50,
        limit: int = 100,
        delay: tuple[float, float] = (1.0, 2.5),
    ) -> None:
        self.keyword = keyword
        self.max_pages = max_pages
        self.sort = sort
        self.likes_min = likes_min
        self.limit = limit
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                "Connection": "keep-alive",
            }
        )
        self.seen: set[str] = set()

    def _choose_headers(self) -> Dict[str, str]:
        return {"User-Agent": random.choice(USER_AGENTS)}

    def fetch_page(self, page: int) -> Optional[str]:
        params = {"keyword": self.keyword, "page": page, "sort": self.sort}
        logging.info("Fetching page %s", page)
        logging.debug("Request params: %s", params)
        try:
            response = self.session.get(
                self.SEARCH_URL,
                params=params,
                headers=self._choose_headers(),
                timeout=15,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            logging.error("Failed to fetch page %s: %s", page, exc)
            return None
        return response.text

    def parse_initial_state(self, html: str) -> Optional[dict]:
        match = INITIAL_STATE_PATTERN.search(html)
        if not match:
            logging.warning("Failed to locate initial state JSON")
            return None
        json_text = match.group(1)
        try:
            data = json.loads(json_text)
            return data
        except json.JSONDecodeError as exc:
            logging.error("Failed to decode initial state JSON: %s", exc)
            return None

    def extract_notes(self, state: dict) -> List[dict]:
        collected: List[dict] = []

        def walk(obj: object) -> None:
            if isinstance(obj, dict):
                if self._looks_like_note(obj):
                    collected.append(obj)
                for value in obj.values():
                    walk(value)
            elif isinstance(obj, list):
                for item in obj:
                    walk(item)

        walk(state)
        logging.debug("Found %d potential notes in initial state", len(collected))
        return collected

    @staticmethod
    def _looks_like_note(candidate: dict) -> bool:
        note_id = candidate.get("note_id") or candidate.get("id")
        title = candidate.get("title") or candidate.get("display_title")
        if not note_id or not title:
            return False
        if candidate.get("note_type") == "video":
            return True  # still return to be filtered later
        return True

    def transform_note(self, data: dict) -> Optional[Note]:
        note_id = str(data.get("note_id") or data.get("id") or "").strip()
        if not note_id:
            return None
        if note_id in self.seen:
            return None

        note_type = str(data.get("note_type") or data.get("type") or "").lower()
        if note_type in {"video", "vlog", "live", "shortvideo", "live_video"}:
            logging.debug("Skipping video note %s", note_id)
            return None

        likes = self._extract_int(
            data,
            keys=["likes", "like_count", "liked_count"],
            interaction_keys=["liked_count", "like_count"],
        )
        if likes is None:
            likes = 0
        if likes < self.likes_min:
            logging.debug("Skipping note %s due to likes=%s", note_id, likes)
            return None

        comments = self._extract_int(
            data,
            keys=["comment_count", "comments"],
            interaction_keys=["comment_count"],
        )
        if comments is None:
            comments = 0

        title = self._extract_first_str(data, ["title", "display_title", "name"])
        desc = self._extract_first_str(data, ["desc", "description", "abstract", "note_content"])

        author_info = data.get("user") or data.get("author") or {}
        author_name = self._extract_first_str(
            author_info, ["nickname", "name", "user_name", "author_name"]
        )
        author_id = self._extract_first_str(
            author_info, ["user_id", "id", "author_id", "uid"]
        )

        release_time = self._extract_first_str(
            data, ["time", "release_time", "publish_time", "last_update_time"]
        )

        images = self._extract_images(data)
        cover_img = self._extract_cover(data, images)

        url = f"{self.BASE_URL}/explore/{note_id}" if note_id else ""

        note = Note(
            keyword=self.keyword,
            note_id=note_id,
            title=title,
            desc=desc,
            author_name=author_name,
            author_id=author_id,
            likes=likes,
            comments=comments,
            release_time=release_time,
            url=url,
            cover_img=cover_img,
            images=images,
        )
        self.seen.add(note_id)
        return note

    @staticmethod
    def _extract_first_str(data: dict, keys: Iterable[str]) -> str:
        for key in keys:
            value = data.get(key)
            if value:
                return str(value).strip()
        return ""

    @staticmethod
    def _extract_int(
        data: dict,
        *,
        keys: Iterable[str],
        interaction_keys: Iterable[str],
    ) -> Optional[int]:
        for key in keys:
            value = data.get(key)
            if isinstance(value, int):
                return value
            if isinstance(value, str) and value.isdigit():
                return int(value)
        interactions = data.get("interactions") or data.get("interaction")
        if isinstance(interactions, dict):
            for key in interaction_keys:
                value = interactions.get(key)
                if isinstance(value, int):
                    return value
                if isinstance(value, str) and value.isdigit():
                    return int(value)
        return None

    @staticmethod
    def _extract_images(data: dict) -> List[str]:
        images: List[str] = []
        image_list = data.get("image_list") or data.get("images") or []
        if isinstance(image_list, list):
            for item in image_list:
                if isinstance(item, dict):
                    url = item.get("url") or item.get("original") or item.get("src")
                    if url:
                        images.append(url)
                elif isinstance(item, str):
                    images.append(item)
        if not images:
            cover = data.get("cover")
            if isinstance(cover, dict):
                url = cover.get("url") or cover.get("original") or cover.get("src")
                if url:
                    images.append(url)
            elif isinstance(cover, str):
                images.append(cover)
        return images

    @staticmethod
    def _extract_cover(data: dict, images: List[str]) -> str:
        cover = data.get("cover")
        if isinstance(cover, dict):
            url = cover.get("url") or cover.get("original") or cover.get("src")
            if url:
                return url
        if isinstance(cover, str):
            return cover
        return images[0] if images else ""

    def scrape(self) -> List[Note]:
        results: List[Note] = []
        for page in range(1, self.max_pages + 1):
            if len(results) >= self.limit:
                break
            html = self.fetch_page(page)
            if not html:
                continue
            state = self.parse_initial_state(html)
            if not state:
                continue
            raw_notes = self.extract_notes(state)
            for raw in raw_notes:
                if len(results) >= self.limit:
                    break
                note = self.transform_note(raw)
                if note:
                    results.append(note)
            sleep_for = random.uniform(*self.delay)
            logging.debug("Sleeping for %.2f seconds", sleep_for)
            time.sleep(sleep_for)
        logging.info("Collected %d notes", len(results))
        return results


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape Xiaohongshu search results")
    parser.add_argument("--keyword", required=True, help="Search keyword")
    parser.add_argument("--pages", type=int, default=5, help="Maximum number of pages to fetch")
    parser.add_argument("--sort", default="popularity", help="Sort order for search results")
    parser.add_argument("--likes-min", type=int, default=50, help="Minimum likes threshold")
    parser.add_argument("--limit", type=int, default=100, help="Maximum number of notes to output")
    parser.add_argument(
        "--out",
        default="output/xhs_posts.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    return parser.parse_args(argv)


def ensure_output_dir(path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def write_to_csv(notes: List[Note], output_path: str) -> None:
    ensure_output_dir(output_path)
    rows = [note.to_row() for note in notes]
    df = pd.DataFrame(rows, columns=[
        "keyword",
        "note_id",
        "title",
        "desc",
        "author_name",
        "author_id",
        "likes",
        "comments",
        "release_time",
        "url",
        "cover_img",
        "images",
    ])
    df.to_csv(output_path, index=False, encoding="utf-8")
    logging.info("Saved %d notes to %s", len(notes), output_path)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    scraper = XiaohongshuScraper(
        keyword=args.keyword,
        max_pages=max(args.pages, 1),
        sort=args.sort,
        likes_min=max(args.likes_min, 0),
        limit=max(args.limit, 1),
    )
    notes = scraper.scrape()
    if not notes:
        logging.warning("No notes collected. Nothing to write.")
        return 1
    write_to_csv(notes, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

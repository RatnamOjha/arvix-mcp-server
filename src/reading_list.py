"""
ReadingList — simple JSON-backed reading list store.
Tracks paper metadata with timestamps and user notes.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

READING_LIST_FILE = "reading_list.json"


class ReadingList:
    def __init__(self, data_dir: Path):
        data_dir.mkdir(parents=True, exist_ok=True)
        self.path = data_dir / READING_LIST_FILE
        self._data: dict[str, dict] = {}
        self._load()

    def add(self, paper_metadata: dict, note: Optional[str] = None) -> None:
        """Add or update a paper in the reading list."""
        arxiv_id = paper_metadata.get("arxiv_id")
        if not arxiv_id:
            return
        entry = {
            **paper_metadata,
            "added_at": datetime.now(timezone.utc).isoformat(),
            "note": note or "",
            "read": False,
        }
        self._data[arxiv_id] = entry
        self._save()
        logger.info(f"Added {arxiv_id} to reading list")

    def mark_read(self, arxiv_id: str) -> bool:
        if arxiv_id in self._data:
            self._data[arxiv_id]["read"] = True
            self._data[arxiv_id]["read_at"] = datetime.now(timezone.utc).isoformat()
            self._save()
            return True
        return False

    def add_note(self, arxiv_id: str, note: str) -> bool:
        if arxiv_id in self._data:
            self._data[arxiv_id]["note"] = note
            self._save()
            return True
        return False

    def remove(self, arxiv_id: str) -> bool:
        if arxiv_id in self._data:
            del self._data[arxiv_id]
            self._save()
            return True
        return False

    def get_all(self) -> list[dict]:
        return sorted(
            list(self._data.values()),
            key=lambda x: x.get("added_at", ""),
            reverse=True,
        )

    def _save(self) -> None:
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2, default=str)

    def _load(self) -> None:
        if self.path.exists():
            with open(self.path) as f:
                self._data = json.load(f)
            logger.info(f"Loaded {len(self._data)} reading list entries")
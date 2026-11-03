"""Raw-file manifest (DQ-12): every raw file with its URL, fetch time and SHA-256.

Raw files are immutable. A re-download writes a new time-stamped file next to the old one and a
new manifest entry; the panel build always reads the latest entry per ``(source, key)``.
Paths in the manifest are POSIX paths relative to the data directory.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MANIFEST_NAME = "manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_stamp(now: datetime | None = None) -> str:
    """Compact UTC stamp for raw file names, e.g. ``20261103T175545Z``."""
    return (now or datetime.now(UTC)).astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def fresh_path(path: Path) -> Path:
    """``path``, or ``stem-1``, ``stem-2``, ... if taken: a raw file is never overwritten."""
    candidate, n = path, 0
    while candidate.exists() or candidate.with_name(candidate.name + ".part").exists():
        n += 1
        candidate = path.with_name(f"{path.stem}-{n}{path.suffix}")
    return candidate


def utc_iso(now: datetime | None = None) -> str:
    return (now or datetime.now(UTC)).astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class ManifestEntry:
    source: str
    key: str
    path: str
    url: str
    fetched_at: str
    sha256: str
    bytes: int
    meta: dict[str, Any] = field(default_factory=dict)


class Manifest:
    """``raw/manifest.json`` under the data directory."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.path = data_dir / "raw" / MANIFEST_NAME
        self.entries: list[ManifestEntry] = []
        if self.path.exists():
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            self.entries = [ManifestEntry(**e) for e in loaded["entries"]]

    def add(
        self,
        source: str,
        key: str,
        file: Path,
        url: str,
        meta: dict[str, Any] | None = None,
        fetched_at: str | None = None,
    ) -> ManifestEntry:
        """Record ``file`` (already written under the data directory) and save the manifest."""
        entry = ManifestEntry(
            source=source,
            key=key,
            path=file.relative_to(self.data_dir).as_posix(),
            url=url,
            fetched_at=fetched_at or utc_iso(),
            sha256=sha256_file(file),
            bytes=file.stat().st_size,
            meta=dict(meta or {}),
        )
        self.entries.append(entry)
        self.save()
        return entry

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        payload = {"entries": [asdict(e) for e in self.entries]}
        tmp.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )
        os.replace(tmp, self.path)

    def latest(self, source: str, key: str) -> ManifestEntry | None:
        """Most recent entry for ``(source, key)``; ties keep the later-added entry."""
        found = [e for e in self.entries if e.source == source and e.key == key]
        if not found:
            return None
        return max(enumerate(found), key=lambda pair: (pair[1].fetched_at, pair[0]))[1]

    def resolve(self, entry: ManifestEntry) -> Path:
        return self.data_dir / entry.path

    def verify(self, entry: ManifestEntry) -> bool:
        """True if the file on disk still has the recorded SHA-256 (raw files are immutable)."""
        path = self.resolve(entry)
        return path.exists() and sha256_file(path) == entry.sha256

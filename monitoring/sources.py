from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable


class LogSource:
    def iter_files(self) -> Iterable[Path]:
        raise NotImplementedError

    def mark_processed(self, path: Path) -> Path:
        raise NotImplementedError


@dataclass
class LocalDirectorySource(LogSource):
    directory: str
    pattern: str = "*.json"
    processed_prefix: str = "_"

    def iter_files(self) -> Generator[Path, None, None]:
        base = Path(self.directory)
        if not base.exists():
            return
        for entry in sorted(base.iterdir()):
            if entry.is_file() and not entry.name.startswith(self.processed_prefix):
                if fnmatch.fnmatch(entry.name, self.pattern):
                    yield entry

    def mark_processed(self, path: Path) -> Path:
        target = path.with_name(f"{self.processed_prefix}{path.name}")
        while target.exists():
            target = target.with_name(f"_{target.name}")
        os.rename(path, target)
        return target

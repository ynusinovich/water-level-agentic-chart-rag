from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

from monitoring.config import get_settings
from monitoring.db import Database
from monitoring.evaluator import RuleBasedEvaluator
from monitoring.parser import parse_log_file
from monitoring.sources import LocalDirectorySource


def process_file(db: Database, evaluator: RuleBasedEvaluator, source: LocalDirectorySource, path, debug: bool = False) -> Optional[int]:
    try:
        rec = parse_log_file(str(path))
        if debug:
            print("[monitoring][debug] file=", path, "station_id=", rec.station_id, "guardrail=", rec.guardrail_triggered)
        log_id = db.insert_log(rec)
        checks = evaluator.evaluate(log_id, rec)
        db.insert_checks(checks)
        source.mark_processed(path)
        return log_id
    except Exception as e:  # pylint: disable=broad-except
        print(f"[monitoring] Failed to process {path}: {e}", file=sys.stderr)
        return None


def run_once(debug: bool = False) -> None:
    settings = get_settings()
    db = Database(settings.database_url)
    db.ensure_schema()
    source = LocalDirectorySource(settings.logs_dir, pattern=settings.file_glob, processed_prefix=settings.processed_prefix)
    evaluator = RuleBasedEvaluator()

    count = 0
    for path in source.iter_files():
        if process_file(db, evaluator, source, path, debug=debug or settings.debug) is not None:
            count += 1
    print(f"[monitoring] Processed {count} file(s)")


def run_watch(debug: bool = False) -> None:
    settings = get_settings()
    db = Database(settings.database_url)
    db.ensure_schema()
    source = LocalDirectorySource(settings.logs_dir, pattern=settings.file_glob, processed_prefix=settings.processed_prefix)
    evaluator = RuleBasedEvaluator()

    print(f"[monitoring] Watching {settings.logs_dir} for {settings.file_glob} (prefix '{settings.processed_prefix}')")
    while True:
        processed_any = False
        for path in source.iter_files():
            if process_file(db, evaluator, source, path, debug=debug or settings.debug) is not None:
                processed_any = True
        if not processed_any:
            time.sleep(settings.poll_seconds)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Monitor logs and store them in Postgres")
    parser.add_argument("--watch", action="store_true", help="Run in watch mode (poll directory)")
    parser.add_argument("--debug", action="store_true", help="Enable debug output during processing")
    args = parser.parse_args(argv)

    if args.watch:
        run_watch(debug=args.debug)
    else:
        run_once(debug=args.debug)


if __name__ == "__main__":
    main()

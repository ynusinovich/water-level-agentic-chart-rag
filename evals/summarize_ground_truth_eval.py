from pathlib import Path
import csv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = PROJECT_ROOT / "evals" / "manual_eval_log.csv"


def main() -> None:
    if not LOG_PATH.exists():
        raise SystemExit(f"manual_eval_log.csv not found at {LOG_PATH}")

    with LOG_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total = len(rows)
    if total == 0:
        print("No rows found in manual_eval_log.csv")
        return

    graded_rows = [r for r in rows if r.get("is_correct", "").strip()]
    num_graded = len(graded_rows)

    def count(label_prefix: str) -> int:
        lp = label_prefix.lower()
        return sum(
            1
            for r in graded_rows
            if r.get("is_correct", "").strip().lower().startswith(lp)
        )

    num_correct = count("correct")
    num_partial = count("partial")
    num_incorrect = count("incorrect")

    print(f"Total questions in log: {total}")
    print(f"Graded (is_correct not empty): {num_graded}")
    print(f"  Correct:   {num_correct}")
    print(f"  Partial:   {num_partial}")
    print(f"  Incorrect: {num_incorrect}")

    if num_graded:
        acc = num_correct / num_graded
        print(f"\nAccuracy (correct / graded): {num_correct}/{num_graded} = {acc:.2%}")


if __name__ == "__main__":
    main()

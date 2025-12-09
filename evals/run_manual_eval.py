"""
Run manual-eval questions against the USGS agent using the unified invoke_agent entrypoint.

- Loads evals/questions.yaml
- Lets you pick a question ID interactively (or via --id)
- Or run all questions at once with --all
- Calls invoke_agent (guardrails + logging) and prints:
    - The full question
    - The agent's answer
    - A summary of tool calls and guardrail status

In --all mode, writes evals/manual_eval_log.csv:
    id, category, question, answer, tool_calls_observed, issues_gaps, follow_ups
"""

import argparse
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional

import yaml
from scripts.invoke_agent import invoke_agent

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = PROJECT_ROOT / "evals" / "questions.yaml"
LOG_PATH = PROJECT_ROOT / "evals" / "manual_eval_log.csv"


def load_questions(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"questions.yaml not found at {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Handle both:
    # - top-level mapping with "questions"
    # - or plain list of question dicts
    if isinstance(data, dict):
        if "questions" in data and isinstance(data["questions"], list):
            return data["questions"]
        else:
            raise ValueError(
                f"Expected key 'questions' with a list, got: {list(data.keys())}"
            )
    elif isinstance(data, list):
        return data

    raise ValueError(
        f"Unexpected questions.yaml structure: {type(data)}; "
        "expected a list or a mapping with key 'questions'."
    )


def choose_question(questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    print("\nAvailable questions:\n")
    for q in questions:
        q_id = q.get("id", "<no-id>")
        text = q.get("question", "").strip()
        short = (text[:80] + "...") if len(text) > 80 else text
        print(f"  {q_id}: {short}")
    print()

    while True:
        choice = input("Enter question ID (or 'q' to quit): ").strip()
        if choice.lower() in {"q", "quit", "exit"}:
            raise SystemExit(0)
        for q in questions:
            if str(q.get("id")) == choice:
                return q
        print(f"ID '{choice}' not found, please try again.")


def format_tool_calls_for_csv(result: Dict[str, Any]) -> str:
    """
    Turn invoke_agent intermediate_steps + guardrail into a single string for CSV.
    """
    steps = result.get("intermediate_steps") or []
    lines = []
    guardrail = result.get("guardrail_reason")
    if guardrail:
        lines.append(f"guardrail={guardrail}")
    for i, step in enumerate(steps, start=1):
        if not isinstance(step, (list, tuple)) or len(step) != 2:
            lines.append(f"[{i}] Unexpected step format: {repr(step)}")
            continue
        action, observation = step
        tool_name = getattr(action, "tool", getattr(action, "tool_name", type(action).__name__))
        tool_input = getattr(action, "tool_input", getattr(action, "input", None))
        lines.append(
            f"[{i}] Tool={tool_name} | input={tool_input!r} | observation={observation!r}"
        )
    return "\n".join(lines)


def run_question(
    question: Dict[str, Any],
) -> Dict[str, str]:
    """
    Run a single question through the agent and print answer + tool calls.

    Returns a dict with keys:
        id, category, question, answer, tool_calls_observed, issues_gaps, follow_ups
    suitable for writing to manual_eval_log.csv.
    """
    q_id = str(question.get("id", "<no-id>"))
    text = question.get("question", "").strip()
    category = str(question.get("category", ""))

    print("\n========================================")
    print(f"Running question {q_id}")
    print("========================================\n")
    print(text)
    print("\n----------------------------------------\n")

    result = invoke_agent(
        text,
        metadata={"source": "manual_eval", "question_id": q_id, "category": category},
    )

    answer_str = str(result.get("output", ""))
    steps = result.get("intermediate_steps", [])

    print("ANSWER:\n")
    print(answer_str)
    print("\n----------------------------------------")
    print("TOOL CALLS (intermediate steps):\n")

    if not steps:
        print("(No intermediate steps recorded.)")
    else:
        for i, step in enumerate(steps, start=1):
            if not isinstance(step, (list, tuple)) or len(step) != 2:
                print(f"[Step {i}] Unexpected step format: {step!r}")
                print("----------------------------------------")
                continue

            action, observation = step
            tool_name = getattr(action, "tool", getattr(action, "tool_name", type(action).__name__))
            tool_input = getattr(action, "tool_input", getattr(action, "input", None))

            print(f"[Step {i}] Tool: {tool_name}")
            print(f"  Tool input: {tool_input!r}")
            print(f"  Observation: {observation!r}")
            print("----------------------------------------")

    print(
        "\nTip: review/edit the CSV row for this question in "
        "evals/manual_eval_log.csv if you ran with --all.\n"
    )

    tool_calls_str = format_tool_calls_for_csv(result)

    return {
        "id": q_id,
        "category": category,
        "question": text,
        "answer": answer_str,
        "tool_calls_observed": tool_calls_str,
        # Leave these blank to annotate later
        "issues_gaps": "",
        "follow_ups": "",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--id",
        help="Run a specific question ID (if omitted and --all is not set, you'll pick from a menu).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Run all questions sequentially (non-interactive) and write "
            "evals/manual_eval_log.csv with id/category/question/answer/tool_calls_observed/issues_gaps/follow_ups."
        ),
    )
    args = parser.parse_args()

    questions = load_questions(QUESTIONS_PATH)
    if args.all:
        # Non-interactive: run every question and write CSV
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "id",
            "category",
            "question",
            "answer",
            "tool_calls_observed",
            "issues_gaps",
            "follow_ups",
        ]
        with LOG_PATH.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for q in questions:
                row = run_question(q)
                writer.writerow(row)
        print(f"\nWrote manual eval log to {LOG_PATH}")
        return

    if args.id:
        selected: Optional[Dict[str, Any]] = None
        for q in questions:
            if str(q.get("id")) == args.id:
                selected = q
                break
        if selected is None:
            all_ids = ", ".join(str(q.get("id")) for q in questions)
            raise SystemExit(
                f"Question ID '{args.id}' not found. Known IDs: {all_ids}"
            )
    else:
        selected = choose_question(questions)

    # For single-question mode, run and print; don't touch the CSV
    run_question(selected)


if __name__ == "__main__":
    main()

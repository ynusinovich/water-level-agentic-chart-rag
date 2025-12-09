"""
Build ground-truth reference answers using the agent (invoke_agent) plus an LLM judge.

Reads evals/manual_eval_log_evaluated.csv (or manual_eval_log_with_gt.csv if present) and writes
evals/manual_eval_log_with_gt.csv with columns:
id, category, question, answer, tool_calls_observed, issues_gaps, follow_ups, reference_answer, is_correct, judge_notes

- Uses invoke_agent(question, metadata={"source": "ground_truth_builder"}) to generate the answer/tool calls.
- Uses OpenAI LLM to generate reference_answer, is_correct, judge_notes with retry/backoff on rate limits.
"""

import csv
import os
import time
import json
from typing import Dict, Any

from openai import OpenAI, RateLimitError

from scripts.invoke_agent import invoke_agent

client = OpenAI()  # uses OPENAI_API_KEY from env

INPUT_CSV = "manual_eval_log_evaluated.csv"
OUTPUT_CSV = "manual_eval_log_with_gt.csv"

SYSTEM_PROMPT = """
You are an evaluator for a water-level agent. 
You are given:

- The user's question.
- The agent's answer.
- The tools it used (if any).
- A human-written analysis of issues/gaps and suggested follow-ups.

Your tasks:

1. Write a concise, factual REFERENCE ANSWER that reflects what a correct,
   ideal answer should look like for this question, based on the information provided.
   Do NOT copy the agent's answer. Improve it.
   If the question clearly cannot be answered with the data available, say so explicitly.

2. Decide if the agent's answer is ACCEPTABLE or NOT, relative to that reference answer.
   Use "yes" if it's good enough for a user (even if not perfect), otherwise "no".

3. Optionally, add short judge notes explaining your decision.

Return JSON with exactly these keys:
- "reference_answer": string
- "is_correct": "yes" or "no"
- "judge_notes": string
"""

def build_messages(row: Dict[str, str]) -> list[Dict[str, Any]]:
    question = row.get("question", "")
    answer = row.get("answer", "")
    tool_calls = row.get("tool_calls_observed", "")
    issues_gaps = row.get("issues_gaps", "")
    follow_ups = row.get("follow_ups", "")

    user_content = f"""
Question:
{question}

Agent answer:
{answer}

Tool calls observed:
{tool_calls}

Issues/gaps (human notes):
{issues_gaps}

Follow-ups (human suggestions):
{follow_ups}
"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content.strip()},
    ]

def call_judge_with_backoff(messages, max_retries: int = 5):
    """Call the chat completion API with rate-limit backoff."""
    delay = 10  # start with 10 seconds
    for attempt in range(1, max_retries + 1):
        try:
            return client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
                messages=messages,
                response_format={"type": "json_object"},
            )
        except RateLimitError as e:
            if attempt == max_retries:
                raise
            print(
                f"Rate limit hit (attempt {attempt}/{max_retries}). "
                f"Sleeping {delay} seconds then retrying..."
            )
            time.sleep(delay)
            delay = min(delay * 2, 60)  # cap at 60 seconds

def judge_row(row: Dict[str, str]) -> Dict[str, str]:
    # If we already have a reference answer, skip re-judging
    if row.get("reference_answer"):
        return row

    # Run the agent through invoke_agent to get answer + tool info
    agent_result = invoke_agent(row.get("question", ""), metadata={"source": "ground_truth_builder", "id": row.get("id")})
    answer = str(agent_result.get("output", ""))
    tool_calls = agent_result.get("intermediate_steps", [])
    tool_calls_str = ""
    if tool_calls:
        lines = []
        for i, step in enumerate(tool_calls, start=1):
            if not isinstance(step, (list, tuple)) or len(step) != 2:
                lines.append(f"[{i}] Unexpected step format: {repr(step)}")
                continue
            action, observation = step
            tool_name = getattr(action, "tool", getattr(action, "tool_name", type(action).__name__))
            tool_input = getattr(action, "tool_input", getattr(action, "input", None))
            lines.append(f"[{i}] Tool={tool_name} | input={tool_input!r} | observation={observation!r}")
        tool_calls_str = "\n".join(lines)
    row["answer"] = answer
    row["tool_calls_observed"] = row.get("tool_calls_observed") or tool_calls_str

    messages = build_messages(row)
    response = call_judge_with_backoff(messages)

    content = response.choices[0].message.content
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # Fallback: if the model somehow returned non-JSON, just store raw
        parsed = {
            "reference_answer": content,
            "is_correct": "no",
            "judge_notes": "Model did not return valid JSON; stored raw content.",
        }

    reference_answer = parsed.get("reference_answer", "").strip()
    is_correct = parsed.get("is_correct", "").strip().lower()
    judge_notes = parsed.get("judge_notes", "").strip()

    # Normalize is_correct to "yes"/"no"
    if is_correct not in ("yes", "no"):
        # crude heuristic
        if "yes" in is_correct:
            is_correct = "yes"
        elif "no" in is_correct:
            is_correct = "no"
        else:
            is_correct = "no"

    row["reference_answer"] = reference_answer
    row["is_correct"] = is_correct
    row["judge_notes"] = judge_notes

    return row

def main():
    input_path = os.path.join("evals", INPUT_CSV) if not os.path.exists(INPUT_CSV) else INPUT_CSV
    output_path = os.path.join("evals", OUTPUT_CSV) if not os.path.exists(OUTPUT_CSV) else OUTPUT_CSV

    with open(input_path, newline="", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = list(reader.fieldnames or [])

        # Make sure we have the new columns
        for col in ["reference_answer", "is_correct", "judge_notes"]:
            if col not in fieldnames:
                fieldnames.append(col)

        rows = list(reader)

    updated_rows = []
    total = len(rows)
    for idx, row in enumerate(rows, start=1):
        print(f"Judging row {idx}/{total}...")
        try:
            updated_row = judge_row(row)
        except RateLimitError as e:
            # This should be rare now, but if it happens after max retries
            print(f"RateLimitError after max retries on row {idx}: {e}")
            raise
        updated_rows.append(updated_row)

    with open(output_path, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)

    print(f"Saved updated ground truth to {output_path}")

if __name__ == "__main__":
    main()

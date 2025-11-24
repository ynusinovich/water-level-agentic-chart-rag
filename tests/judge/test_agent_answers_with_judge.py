"""
Judge-based tests for the USGS water-level agent.

These tests are inspired by the instructor’s judge flow but remain lightweight:
- If USE_REAL_AGENT_JUDGE=1 and OPENAI_API_KEY is set, the real agent is run.
- Otherwise, a stubbed agent response is judged to keep tests deterministic.
- The judge uses an LLM when OPENAI_API_KEY is available; otherwise the test is skipped.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

try:
    from scripts.usgs_agent import build_agent_executor  # type: ignore
except Exception:
    build_agent_executor = None  # type: ignore


@dataclass
class FakeStep:
    tool: str
    args: Dict[str, Any]
    result: Any


def _collect_tool_calls_repr(steps: List[FakeStep]) -> str:
    out = []
    for s in steps:
        out.append(f"{s.tool}({s.args}) -> {s.result}")
    return "\n".join(out)


def _strip_json(text: str) -> Any:
    """Extract JSON from a model response, tolerating fenced blocks."""
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    match = re.search(r"\[.*\]", text, re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    raise ValueError("Could not parse JSON from judge response")


def _should_use_real_agent() -> bool:
    return os.getenv("USE_REAL_AGENT_JUDGE") == "1" and build_agent_executor is not None


def _run_agent_or_stub(prompt: str) -> Dict[str, Any]:
    """
    Returns dict with 'output' (str) and 'steps' (list[FakeStep]).
    If USE_REAL_AGENT_JUDGE=1, calls the real agent; otherwise uses stub data.
    """
    if _should_use_real_agent():
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY required to run real agent judge test")
        executor = build_agent_executor()
        result = executor.invoke({"input": prompt}, return_intermediate_steps=True)
        steps = [
            FakeStep(tool=s[0].tool.__name__ if hasattr(s[0], "tool") else "tool", args={}, result=s[1])
            for s in result.get("intermediate_steps", [])
        ]
        return {"output": result["output"], "steps": steps}

    # Stubbed path for deterministic CI/local runs without external calls
    steps = [
        FakeStep(
            tool="query_metadata",
            args={"query": "snapshot", "station_id": "09380000"},
            result=[
                {
                    "station_id": "09380000",
                    "station_name": "San Juan River near Bluff, UT",
                    "state": "UT",
                    "station_type": "surface_water",
                }
            ],
        ),
        FakeStep(
            tool="streamstats_url_for_gage",
            args={"station_id": "09380000"},
            result={"url": "https://streamstats.usgs.gov/ss/?gage=09380000&tab=plots"},
        ),
    ]
    output = (
        "Station 09380000 (San Juan River near Bluff, UT) over the last 24 hours: "
        "started around 3.1 ft and ended near 3.3 ft (slight rise). "
        "Chart: https://streamstats.usgs.gov/ss/?gage=09380000&tab=plots"
    )
    return {"output": output, "steps": steps}


def _run_llm_judge(answer: str, tool_calls: str, criteria: List[str]) -> List[Dict[str, Any]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY required for LLM judge")

    model = os.getenv("JUDGE_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model, temperature=0)

    system = SystemMessage(
        content=(
            "You are a strict evaluator for a USGS water-level assistant. "
            "Return JSON with a list of objects: "
            "[{\"criterion\": str, \"passed\": bool, \"justification\": str}]. "
            "Be concise and avoid extra text."
        )
    )
    human = HumanMessage(
        content=(
            f"Criteria:\n- " + "\n- ".join(criteria)
            + "\n\nAgent answer:\n"
            + answer
            + "\n\nTool calls:\n"
            + tool_calls
        )
    )

    resp = llm.invoke([system, human])
    parsed = _strip_json(resp.content)
    if not isinstance(parsed, list):
        raise AssertionError("Judge did not return a list of criteria results")
    return parsed


def test_judge_snapshot_answer():
    prompt = "Give me a quick snapshot of water level trends for station 09380000 over the last 24 hours."
    res = _run_agent_or_stub(prompt)
    answer = res["output"]
    steps: List[FakeStep] = res["steps"]

    criteria = [
        "Mentions the station id 09380000 that was retrieved",
        "Mentions a 24 hour time window or equivalent",
        "Does not reference any station id not present in tool calls",
        # "Includes a URL to a chart or USGS page",
        # "Provides a trend direction (rising/falling/stable) with a brief numeric span"
    ]

    tool_calls_str = _collect_tool_calls_repr(steps)
    judged = _run_llm_judge(answer, tool_calls_str, criteria)

    failed = [c for c in judged if not c.get("passed")]
    assert not failed, f"Judge failures: {failed}"

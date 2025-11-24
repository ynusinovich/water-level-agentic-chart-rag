"""
Lightweight structure checks for the USGS water-level agent.

These tests avoid real network/Qdrant/OpenAI calls by using a stub executor
that mimics the agent interface (invoke with {"input": ...}, returns an
output string plus intermediate_steps describing tool calls).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List

import pytest


@dataclass
class FakeStep:
    """Simple container for a tool call and its return value."""

    tool: str
    args: Dict[str, Any]
    result: Any


class StubExecutor:
    """
    Mimics the agent executor interface from scripts.usgs_agent.
    Uses pre-seeded steps/output for deterministic assertions.
    """

    def __init__(self, steps: List[FakeStep], output: str):
        self.steps = steps
        self.output = output

    def invoke(self, params: Dict[str, Any], return_intermediate_steps: bool = False):
        if return_intermediate_steps:
            return {"output": self.output, "intermediate_steps": self.steps}
        return {"output": self.output}


def _station_ids_from_steps(steps: List[FakeStep]) -> List[str]:
    """Extract station_ids from a fake query_metadata tool result."""
    ids = []
    for step in steps:
        if step.tool != "query_metadata":
            continue
        for item in step.result:
            sid = item.get("station_id")
            if sid:
                ids.append(str(sid))
    return ids


def _has_url(text: str) -> bool:
    return bool(re.search(r"https?://", text))


@pytest.fixture()
def stub_executor_snapshot() -> StubExecutor:
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
                    "description": "Test description",
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
    return StubExecutor(steps=steps, output=output)


@pytest.fixture()
def stub_executor_no_data() -> StubExecutor:
    steps = [
        FakeStep(
            tool="query_metadata",
            args={"query": "missing", "station_id": "99999999"},
            result=[],
        )
    ]
    output = (
        "Could not find recent data for station 99999999. "
        "The station was not in the index or has no active chart."
    )
    return StubExecutor(steps=steps, output=output)


def test_snapshot_response_mentions_retrieved_station_and_url(stub_executor_snapshot: StubExecutor):
    query = "Give me a quick snapshot of water level trends for station 09380000 over the last 24 hours."
    result = stub_executor_snapshot.invoke({"input": query}, return_intermediate_steps=True)

    output = result["output"]
    steps: List[FakeStep] = result["intermediate_steps"]

    # Non-empty output
    assert output.strip(), "Expected non-empty agent output"

    # Station ID mentioned and was actually retrieved
    retrieved_ids = _station_ids_from_steps(steps)
    assert retrieved_ids, "Expected at least one station_id from retrieval"
    assert any(sid in output for sid in retrieved_ids), "Output should reference a retrieved station_id"

    # Time window echoed back (24 hours)
    assert "24" in output and ("hour" in output.lower() or "hr" in output.lower()), "Expected a time window mention"

    # URL or chart reference included
    assert _has_url(output), "Expected a chart/USGS URL in the output"


def test_handles_missing_station_gracefully(stub_executor_no_data: StubExecutor):
    query = "Give me a quick snapshot of water level trends for station 99999999 over the last 24 hours."
    result = stub_executor_no_data.invoke({"input": query}, return_intermediate_steps=True)

    output = result["output"]
    steps: List[FakeStep] = result["intermediate_steps"]

    # Output should acknowledge absence, not hallucinate data
    assert "no" in output.lower() or "could not" in output.lower(), "Expected explicit mention of missing data"
    assert "99999999" in output, "Should echo the requested station ID"

    retrieved_ids = _station_ids_from_steps(steps)
    assert retrieved_ids == [], "Stubbed retrieval returned no stations"
    assert not _has_url(output), "Should not invent a chart URL when none was found"

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class CheckName(str, Enum):
    station_present = "station_present"
    chart_extracted = "chart_extracted"
    numeric_grounded = "numeric_grounded"
    guardrail_not_triggered = "guardrail_not_triggered"
    answer_mentions_station = "answer_mentions_station"


@dataclass
class LLMLogRecord:
    filepath: str
    agent_name: Optional[str]
    provider: Optional[str]
    model: Optional[str]
    user_prompt: Optional[str]
    station_id: Optional[str]
    state_filter: Optional[str]
    time_window: Optional[str]
    question_type: Optional[str]
    guardrail_triggered: bool
    guardrail_reason: Optional[str]
    total_input_tokens: Optional[int]
    total_output_tokens: Optional[int]
    assistant_answer: Optional[str]
    tools_json: Optional[str]
    raw_json: Optional[str]


@dataclass
class CheckResult:
    log_id: int
    check_name: CheckName
    passed: Optional[bool] = None
    score: Optional[float] = None
    details: Optional[str] = None


@dataclass
class Feedback:
    log_id: int
    is_good: bool
    comments: Optional[str] = None
    reference_answer: Optional[str] = None

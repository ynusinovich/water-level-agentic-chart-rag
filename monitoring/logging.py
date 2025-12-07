from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ToolEvent:
    name: str
    input: Any
    output: Any = None
    error: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None


@dataclass
class LogRecord:
    request_id: str
    created_at: str
    user_prompt: str
    station_ids: List[str]
    state_filter: Optional[str]
    time_window: Optional[str]
    question_type: Optional[str]
    guardrail_triggered: bool
    guardrail_reason: Optional[str]
    model: Optional[str]
    provider: Optional[str]
    tokens_input: Optional[int]
    tokens_output: Optional[int]
    tools: List[ToolEvent]
    answer_text: Optional[str]
    errors: List[str] = field(default_factory=list)
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def build_log_record(
    user_prompt: str,
    tools: List[ToolEvent],
    answer_text: Optional[str],
    *,
    request_id: Optional[str] = None,
    station_ids: Optional[List[str]] = None,
    state_filter: Optional[str] = None,
    time_window: Optional[str] = None,
    question_type: Optional[str] = None,
    guardrail_triggered: bool = False,
    guardrail_reason: Optional[str] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    tokens_input: Optional[int] = None,
    tokens_output: Optional[int] = None,
    errors: Optional[List[str]] = None,
    source: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> LogRecord:
    return LogRecord(
        request_id=request_id or str(uuid.uuid4()),
        created_at=datetime.utcnow().isoformat(),
        user_prompt=user_prompt,
        station_ids=station_ids or [],
        state_filter=state_filter,
        time_window=time_window,
        question_type=question_type,
        guardrail_triggered=guardrail_triggered,
        guardrail_reason=guardrail_reason,
        model=model,
        provider=provider,
        tokens_input=tokens_input,
        tokens_output=tokens_output,
        tools=tools,
        answer_text=answer_text,
        errors=errors or [],
        source=source,
        metadata=metadata or {},
    )


def write_json_log(record: LogRecord, logs_dir: str = "logs") -> Path:
    base = Path(logs_dir)
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{record.request_id}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            asdict(record),
            f,
            indent=2,
            default=str,
        )
    return path

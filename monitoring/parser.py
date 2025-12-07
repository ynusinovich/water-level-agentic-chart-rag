from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from .schemas import LLMLogRecord


def _first_station_id(doc: Dict[str, Any]) -> Optional[str]:
    ids = doc.get("station_ids") or []
    if isinstance(ids, list) and ids:
        return str(ids[0])
    return None


def _tool_summary(doc: Dict[str, Any]) -> str:
    tools = doc.get("tools") or []
    try:
        return json.dumps(tools)
    except Exception:
        return ""


def parse_log_file(path: str | Path) -> LLMLogRecord:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = f.read()
    doc = json.loads(raw)

    return LLMLogRecord(
        filepath=str(p),
        agent_name=doc.get("agent_name"),
        provider=doc.get("provider"),
        model=doc.get("model"),
        user_prompt=doc.get("user_prompt"),
        station_id=_first_station_id(doc),
        state_filter=doc.get("state_filter"),
        time_window=doc.get("time_window"),
        question_type=doc.get("question_type"),
        guardrail_triggered=bool(doc.get("guardrail_triggered")),
        guardrail_reason=doc.get("guardrail_reason"),
        total_input_tokens=doc.get("tokens_input"),
        total_output_tokens=doc.get("tokens_output"),
        assistant_answer=doc.get("answer_text"),
        tools_json=_tool_summary(doc),
        raw_json=raw,
    )

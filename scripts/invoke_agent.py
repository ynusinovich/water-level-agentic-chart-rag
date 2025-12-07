from __future__ import annotations

import os
from typing import Any, Dict, Optional
from uuid import uuid4

from monitoring.callbacks import CallbackState, MonitoringCallback
from scripts.guardrails import (
    GuardrailResult,
    OutputGuardrailResult,
    apply_output_guardrails,
    validate_input,
)


def _detect_question_type(user_text: str) -> Optional[str]:
    text = user_text.lower()
    if "trend" in text:
        return "trend"
    if "snapshot" in text or "last 24" in text:
        return "snapshot"
    if "compare" in text:
        return "comparison"
    return None


def invoke_agent(user_text: str, metadata: Optional[Dict[str, Any]] = None, logs_dir: str = "logs"):
    """
    Unified entrypoint that applies guardrails, runs the agent with monitoring callbacks,
    and writes structured logs.
    """
    from scripts.usgs_agent import build_agent_executor  # local import to avoid cycles

    request_id = str(uuid4())
    metadata = metadata or {}
    input_check: GuardrailResult = validate_input(user_text)
    question_type = _detect_question_type(user_text)

    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    provider = "openai"
    callback_state = CallbackState(
        user_prompt=user_text,
        request_id=request_id,
        logs_dir=logs_dir,
        station_ids=input_check.station_ids,
        state_filter=input_check.state_filter,
        time_window=input_check.time_window,
        question_type=question_type,
        model=model_name,
        provider=provider,
        metadata=metadata,
        source=metadata.get("source"),
    )
    cb = MonitoringCallback(callback_state)

    if input_check.fail:
        blocked_msg = input_check.user_message or "Request blocked by guardrail."
        cb.finalize(
            answer_text=blocked_msg,
            guardrail_triggered=True,
            guardrail_reason=input_check.reason,
        )
        return {
            "output": blocked_msg,
            "intermediate_steps": [],
            "guardrail_triggered": True,
            "guardrail_reason": input_check.reason,
        }

    executor = build_agent_executor()
    try:
        executor.return_intermediate_steps = True  # type: ignore[attr-defined]
    except Exception:
        pass

    lc_config = {
        "callbacks": [cb],
        "tags": [
            "usgs-agent",
            f"request:{request_id}",
            f"station:{input_check.station_ids[0]}" if input_check.station_ids else "station:unknown",
            f"guardrail:{'soft' if input_check.soft_flag else 'none'}",
        ],
        "metadata": {
            "station_ids": input_check.station_ids,
            "question_type": question_type,
            "guardrail_soft": input_check.soft_flag,
            "time_window": input_check.time_window,
            "state_filter": input_check.state_filter,
        },
    }

    result = executor.invoke({"input": user_text}, config=lc_config)
    intermediate_steps = result.get("intermediate_steps", [])
    raw_answer = result.get("output", result)

    og: OutputGuardrailResult = apply_output_guardrails(
        answer=str(raw_answer),
        intermediate_steps=intermediate_steps,
        station_ids=input_check.station_ids,
    )

    cb.finalize(
        answer_text=og.answer,
        guardrail_triggered=og.triggered,
        guardrail_reason=og.reason,
    )

    return {
        "output": og.answer,
        "intermediate_steps": intermediate_steps,
        "guardrail_triggered": og.triggered,
        "guardrail_reason": og.reason,
    }

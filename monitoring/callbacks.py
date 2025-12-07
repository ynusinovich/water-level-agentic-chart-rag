from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from .logging import ToolEvent, build_log_record, write_json_log


@dataclass
class CallbackState:
    user_prompt: str
    request_id: str
    logs_dir: str
    station_ids: List[str]
    state_filter: Optional[str]
    time_window: Optional[str]
    question_type: Optional[str]
    model: Optional[str]
    provider: Optional[str]
    guardrail_triggered: bool = False
    guardrail_reason: Optional[str] = None
    tools: List[ToolEvent] = None
    errors: List[str] = None
    tokens_input: Optional[int] = None
    tokens_output: Optional[int] = None
    metadata: Dict[str, Any] = None
    source: Optional[str] = None

    def __post_init__(self):
        self.tools = self.tools or []
        self.errors = self.errors or []
        self.metadata = self.metadata or {}


class MonitoringCallback(BaseCallbackHandler):
    """Collects tool/LLM events and writes a structured log at the end."""

    def __init__(self, state: CallbackState):
        self.state = state
        self._current_action: Optional[ToolEvent] = None

    # Tool lifecycle
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, run_id, parent_run_id=None, **kwargs):
        name = serialized.get("name") or serialized.get("id") or "tool"
        self._current_action = ToolEvent(name=name, input=input_str)

    def on_tool_end(self, output: Any, run_id, parent_run_id=None, **kwargs):
        if self._current_action:
            self._current_action.output = output
            self.state.tools.append(self._current_action)
        self._current_action = None

    def on_tool_error(self, error: BaseException, run_id, parent_run_id=None, **kwargs):
        if self._current_action:
            self._current_action.error = str(error)
            self.state.tools.append(self._current_action)
        else:
            self.state.errors.append(str(error))
        self._current_action = None

    # LLM usage
    def on_llm_end(self, response: LLMResult, **kwargs):
        usage = (response.llm_output or {}).get("token_usage") or {}
        self.state.tokens_input = (self.state.tokens_input or 0) + int(usage.get("prompt_tokens") or 0)
        self.state.tokens_output = (self.state.tokens_output or 0) + int(usage.get("completion_tokens") or 0)

    # Agent errors
    def on_chain_error(self, error: BaseException, **kwargs):
        self.state.errors.append(str(error))

    # Finalize log (called manually by invoke_agent)
    def finalize(self, answer_text: Optional[str], guardrail_triggered: bool = False, guardrail_reason: Optional[str] = None):
        self.state.guardrail_triggered = guardrail_triggered or self.state.guardrail_triggered
        self.state.guardrail_reason = guardrail_reason or self.state.guardrail_reason

        record = build_log_record(
            user_prompt=self.state.user_prompt,
            tools=self.state.tools,
            answer_text=answer_text,
            request_id=self.state.request_id,
            station_ids=self.state.station_ids,
            state_filter=self.state.state_filter,
            time_window=self.state.time_window,
            question_type=self.state.question_type,
            guardrail_triggered=self.state.guardrail_triggered,
            guardrail_reason=self.state.guardrail_reason,
            model=self.state.model,
            provider=self.state.provider,
            tokens_input=self.state.tokens_input,
            tokens_output=self.state.tokens_output,
            errors=self.state.errors,
            source=self.state.source,
            metadata=self.state.metadata,
        )
        return write_json_log(record, logs_dir=self.state.logs_dir)

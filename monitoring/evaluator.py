from __future__ import annotations

import json
import re
from typing import List

from .schemas import CheckName, CheckResult, LLMLogRecord


class RuleBasedEvaluator:
    """Simple, deterministic checks for monitoring dashboards."""

    def evaluate(self, log_id: int, record: LLMLogRecord) -> List[CheckResult]:
        checks: List[CheckResult] = []
        answer = record.assistant_answer or ""

        # station present in retrieval or prompt
        station_present = bool(record.station_id) or bool(re.search(r"\b\d{7,15}\b", record.user_prompt or ""))
        checks.append(
            CheckResult(
                log_id=log_id,
                check_name=CheckName.station_present,
                passed=station_present,
                details=f"station_id={record.station_id}",
            )
        )

        # chart extracted success flag from tools_json
        chart_extracted = False
        try:
            tools = json.loads(record.tools_json or "[]")
            for t in tools:
                if isinstance(t, dict):
                    name = t.get("name") or ""
                    if name in {"extract_highcharts_series", "extract_plotly_series"}:
                        out = t.get("output")
                        if isinstance(out, dict) and out.get("success"):
                            chart_extracted = True
                        elif isinstance(out, str) and '"success": true' in out.lower():
                            chart_extracted = True
        except Exception:
            pass
        checks.append(
            CheckResult(
                log_id=log_id,
                check_name=CheckName.chart_extracted,
                passed=chart_extracted,
                details="chart extraction success" if chart_extracted else "no successful chart extract",
            )
        )

        # numeric grounded only if chart extracted
        has_numeric = bool(re.search(r"\d", answer))
        checks.append(
            CheckResult(
                log_id=log_id,
                check_name=CheckName.numeric_grounded,
                passed=(not has_numeric) or chart_extracted,
                details=f"has_numeric={has_numeric}, chart_extracted={chart_extracted}",
            )
        )

        # guardrail not triggered
        checks.append(
            CheckResult(
                log_id=log_id,
                check_name=CheckName.guardrail_not_triggered,
                passed=(record.guardrail_triggered is False),
                details=record.guardrail_reason or "",
            )
        )

        # answer mentions station id
        mentions_station = bool(record.station_id and record.station_id in answer)
        checks.append(
            CheckResult(
                log_id=log_id,
                check_name=CheckName.answer_mentions_station,
                passed=mentions_station if record.station_id else None,
                details=f"station_id={record.station_id}",
            )
        )

        return checks

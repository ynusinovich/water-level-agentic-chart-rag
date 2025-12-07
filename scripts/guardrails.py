from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class GuardrailResult:
    fail: bool
    reason: Optional[str] = None
    user_message: Optional[str] = None
    soft_flag: bool = False
    station_ids: List[str] = []
    state_filter: Optional[str] = None
    time_window: Optional[str] = None
    question_type: Optional[str] = None

    def __post_init__(self):
        self.station_ids = self.station_ids or []


@dataclass
class OutputGuardrailResult:
    triggered: bool
    reason: Optional[str] = None
    answer: str = ""


OFF_TOPIC_HINTS = {"love poem", "romantic", "recipe", "song lyrics", "movie", "book review"}
ALLOWED_STATE_CODES = {"AZ", "CA", "CO", "NM", "UT", "NV"}

# Basic US state name/abbrev map for detection
STATE_NAME_TO_CODE = {
    "AL": "AL", "ALABAMA": "AL",
    "AK": "AK", "ALASKA": "AK",
    "AZ": "AZ", "ARIZONA": "AZ",
    "AR": "AR", "ARKANSAS": "AR",
    "CA": "CA", "CALIFORNIA": "CA",
    "CO": "CO", "COLORADO": "CO",
    "CT": "CT", "CONNECTICUT": "CT",
    "DE": "DE", "DELAWARE": "DE",
    "FL": "FL", "FLORIDA": "FL",
    "GA": "GA", "GEORGIA": "GA",
    "HI": "HI", "HAWAII": "HI",
    "ID": "ID", "IDAHO": "ID",
    "IL": "IL", "ILLINOIS": "IL",
    "IN": "IN", "INDIANA": "IN",
    "IA": "IA", "IOWA": "IA",
    "KS": "KS", "KANSAS": "KS",
    "KY": "KY", "KENTUCKY": "KY",
    "LA": "LA", "LOUISIANA": "LA",
    "ME": "ME", "MAINE": "ME",
    "MD": "MD", "MARYLAND": "MD",
    "MA": "MA", "MASSACHUSETTS": "MA",
    "MI": "MI", "MICHIGAN": "MI",
    "MN": "MN", "MINNESOTA": "MN",
    "MS": "MS", "MISSISSIPPI": "MS",
    "MO": "MO", "MISSOURI": "MO",
    "MT": "MT", "MONTANA": "MT",
    "NE": "NE", "NEBRASKA": "NE",
    "NV": "NV", "NEVADA": "NV",
    "NH": "NH", "NEW HAMPSHIRE": "NH",
    "NJ": "NJ", "NEW JERSEY": "NJ",
    "NM": "NM", "NEW MEXICO": "NM",
    "NY": "NY", "NEW YORK": "NY",
    "NC": "NC", "NORTH CAROLINA": "NC",
    "ND": "ND", "NORTH DAKOTA": "ND",
    "OH": "OH", "OHIO": "OH",
    "OK": "OK", "OKLAHOMA": "OK",
    "OR": "OR", "OREGON": "OR",
    "PA": "PA", "PENNSYLVANIA": "PA",
    "RI": "RI", "RHODE ISLAND": "RI",
    "SC": "SC", "SOUTH CAROLINA": "SC",
    "SD": "SD", "SOUTH DAKOTA": "SD",
    "TN": "TN", "TENNESSEE": "TN",
    "TX": "TX", "TEXAS": "TX",
    "UT": "UT", "UTAH": "UT",
    "VT": "VT", "VERMONT": "VT",
    "VA": "VA", "VIRGINIA": "VA",
    "WA": "WA", "WASHINGTON": "WA",
    "WV": "WV", "WEST VIRGINIA": "WV",
    "WI": "WI", "WISCONSIN": "WI",
    "WY": "WY", "WYOMING": "WY",
}


def _parse_station_ids(text: str) -> List[str]:
    """
    Extract candidate station id tokens from the text.

    Allow 7-15 character alphanumeric/hyphen tokens that contain at least
    one digit. Purely numeric tokens are treated as valid USGS IDs; mixed
    tokens (letters or hyphens) will later be rejected as invalid_station_id.
    """
    ids: List[str] = []
    # Allow letters, digits, and hyphens, 7–15 chars
    for m in re.finditer(r"\b[A-Za-z0-9\-]{7,15}\b", text):
        token = m.group(0)
        # Require at least one digit so we don't treat normal words as IDs
        if not any(ch.isdigit() for ch in token):
            continue
        if token not in ids:
            ids.append(token)
    return ids


def _parse_state(text: str) -> tuple[Optional[str], Optional[bool]]:
    """
    Returns (state_code, allowed_flag). allowed_flag is True/False when a state is detected, None if none.

    Rules:
      - Full state names are matched case-insensitively for the states we actually support.
      - 2-letter codes are only considered if they appear in ALL CAPS (e.g., 'AZ', 'NM'),
        so 'me' in normal text won't be treated as 'ME' (Maine).
    """
    # 1) Full names for the six states we actually ingest
    lower = text.lower()
    for name, code in (
        ("arizona", "AZ"),
        ("california", "CA"),
        ("colorado", "CO"),
        ("new mexico", "NM"),
        ("utah", "UT"),
        ("nevada", "NV"),
    ):
        if re.search(rf"\b{name}\b", lower):
            return code, code in ALLOWED_STATE_CODES

    # 2) Two-letter codes, only when written in ALL CAPS
    for tok in re.findall(r"\b[A-Z]{2}\b", text):
        # STATE_NAME_TO_CODE already maps codes like 'AZ', 'CA', 'TX', etc. to their canonical form
        if tok in STATE_NAME_TO_CODE:
            code = STATE_NAME_TO_CODE[tok]
            return code, code in ALLOWED_STATE_CODES

    return None, None


def _parse_time_window(text: str) -> Optional[str]:
    m = re.search(r"last\s+(\d+)\s*(hour|hours|day|days|week|weeks|month|months|year|years)", text, re.I)
    if not m:
        return None
    value = int(m.group(1))
    unit = m.group(2).lower()
    if value < 0:
        return "invalid"
    if unit.startswith("year") and value > 1:
        return "too_long"
    if unit.startswith("month") and value > 12:
        return "too_long"
    if unit.startswith("week") and value > 52:
        return "too_long"
    if unit.startswith("day") and value > 365:
        return "too_long"
    if unit.startswith("hour") and value > 24 * 365:
        return "too_long"
    return f"{value} {unit}"


def validate_input(user_text: str) -> GuardrailResult:
    text = user_text.strip()
    station_ids = _parse_station_ids(text)
    time_window = _parse_time_window(text)
    state_code, state_allowed = _parse_state(text)

    lower = text.lower()
    if any(hint in lower for hint in OFF_TOPIC_HINTS):
        return GuardrailResult(
            fail=True,
            reason="off_topic",
            user_message="This assistant focuses on USGS water data questions.",
            station_ids=station_ids,
        )

    if time_window in {"invalid", "too_long"}:
        return GuardrailResult(
            fail=True,
            reason="invalid_time_window",
            user_message="Please use a recent, reasonable time window (e.g., last 24 hours or 7 days).",
            station_ids=station_ids,
        )

    if state_code and state_allowed is False:
        return GuardrailResult(
            fail=True,
            reason="state_not_supported",
            user_message="I only have data for AZ, CA, CO, NM, UT, and NV. Please ask about one of those states.",
            station_ids=station_ids,
        )

    if station_ids and not all(s.isdigit() for s in station_ids):
        return GuardrailResult(
            fail=True,
            reason="invalid_station_id",
            user_message="Station IDs must be digits only (7-15 characters).",
            station_ids=[],
        )

    soft_flag = not station_ids
    return GuardrailResult(
        fail=False,
        soft_flag=soft_flag,
        station_ids=station_ids,
        time_window=None if time_window in {"invalid", "too_long"} else time_window,
        state_filter=state_code if state_allowed else None,
    )


def _extract_successful_series(step_output: Any) -> bool:
    if isinstance(step_output, dict):
        return bool(step_output.get("success"))
    if isinstance(step_output, str):
        low = step_output.lower()
        return '"success": true' in low or "'success': true" in low
    return False


def apply_output_guardrails(answer: str, intermediate_steps: list, station_ids: List[str]) -> OutputGuardrailResult:
    has_extract_success = False
    for step in intermediate_steps or []:
        if not isinstance(step, (list, tuple)) or len(step) != 2:
            continue
        action, observation = step
        tool_name = None
        if isinstance(action, dict):
            tool_name = action.get("tool")
        elif hasattr(action, "tool"):
            tool_name = getattr(action, "tool", None)
        if tool_name and tool_name not in {"extract_highcharts_series", "extract_plotly_series"}:
            continue
        if _extract_successful_series(observation):
            has_extract_success = True
            break

    if has_extract_success:
        return OutputGuardrailResult(triggered=False, answer=answer)

    station = station_ids[0] if station_ids else "the requested station"
    disclaimer = (
        f"No recent chart data was available for {station}; numeric values are omitted. "
        "Please provide another station or timeframe."
    )
    if answer:
        adjusted = f"{disclaimer}\n\nPrevious draft (for transparency, may be incomplete):\n{answer}"
    else:
        adjusted = disclaimer
    return OutputGuardrailResult(
        triggered=True,
        reason="no_chart_data",
        answer=adjusted,
    )

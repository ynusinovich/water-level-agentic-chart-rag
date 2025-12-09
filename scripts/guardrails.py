from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class GuardrailResult:
    fail: bool
    reason: Optional[str] = None
    user_message: Optional[str] = None
    soft_flag: bool = False
    station_ids: Optional[List[str]] = None
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
    warnings: Optional[str] = None


OFF_TOPIC_HINTS = {"love poem", "romantic", "recipe", "song lyrics", "movie", "book review"}
ALLOWED_STATE_CODES = {"AZ", "CA", "CO", "NM", "UT", "NV"}
STATION_ID_REGEX = re.compile(r"\d{7,15}")

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

    Find 7-15 digit tokens (purely numeric). Mixed alnum tokens are handled
    separately as invalids.
    """
    ids: List[str] = []
    for m in STATION_ID_REGEX.finditer(text):
        token = m.group(0)
        if token not in ids:
            ids.append(token)
    return ids


def _find_mixed_station_tokens(text: str) -> List[str]:
    """
    Find 7-15 length tokens that include digits but are not all digits
    (likely invalid station ids such as 'ABC123XYZ').
    """
    mixed: List[str] = []
    for m in re.finditer(r"\b[A-Za-z0-9\-]{7,15}\b", text):
        tok = m.group(0)
        if any(ch.isdigit() for ch in tok) and not tok.isdigit():
            mixed.append(tok)
    return mixed


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
    """
    Parse user text for a time window like:
      - 'last 24 hours', 'last 7 days', 'last 2 weeks', ...
      - or phrase-only: 'last week', 'last month', 'last year', 'last decade'

    Returns a normalized string like '24 hours', '7 days', '30 days', '365 days', '10 years',
    or 'invalid' / 'too_long' / None.
    """
    if not text:
        return None

    lower = text.lower()

    # Handle phrase-only windows (no explicit number)
    if re.search(r"\blast\s+week\b", lower):
        return "7 days"
    if re.search(r"\blast\s+month\b", lower):
        return "30 days"
    if re.search(r"\blast\s+year\b", lower):
        return "365 days"
    if re.search(r"\blast\s+decade\b", lower):
        # Treat 'last decade' as a 10-year window
        return "10 years"

    # Existing numeric pattern: 'last 24 hours', 'last 7 days', etc.
    m = re.search(
        r"last\s+(\d+)\s*(hour|hours|day|days|week|weeks|month|months|year|years)",
        lower,
    )
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
    mixed_station_tokens = _find_mixed_station_tokens(text)
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
    if not station_ids and mixed_station_tokens:
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


def _extract_series_values(step_output: Any) -> list[float]:
    vals = []
    data = None
    if isinstance(step_output, dict):
        data = step_output.get("data") if "data" in step_output else step_output.get("values")
        if data is None and "output" in step_output:
            data = step_output.get("output")
    elif isinstance(step_output, str):
        try:
            parsed = json.loads(step_output)
            data = parsed.get("data")
        except Exception:
            data = None
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "y" in item:
                try:
                    vals.append(float(item["y"]))
                except Exception:
                    continue
            else:
                try:
                    vals.append(float(item))
                except Exception:
                    continue
    return vals


def _extract_timestamps_ms(step_output: Any) -> list[float]:
    parsed = step_output
    if isinstance(step_output, str):
        try:
            parsed = json.loads(step_output)
        except Exception:
            return []
    if not isinstance(parsed, dict):
        return []
    data = parsed.get("data") if "data" in parsed else parsed.get("values")
    if not isinstance(data, list) or not data:
        return []
    xs: list[float] = []
    for item in data:
        if isinstance(item, dict) and "x" in item:
            try:
                xs.append(float(item.get("x")))
            except Exception:
                continue
    return xs


def _parse_time_window_to_timedelta(time_window: Optional[str]) -> Optional[dt.timedelta]:
    """
    Convert normalized time_window strings like '24 hours', '48 hours', '7 days',
    '2 weeks', '6 months', '1 year', '10 years' into a timedelta.
    Returns None if parsing fails.
    """
    if not time_window:
        return None
    win = time_window.strip().lower()
    m = re.search(r"(\d+)\s*(hour|hours|day|days|week|weeks|month|months|year|years|decade|decades)", win)
    if not m:
        return None
    val = int(m.group(1))
    unit = m.group(2)
    if "hour" in unit:
        return dt.timedelta(hours=val)
    if "day" in unit:
        return dt.timedelta(days=val)
    if "week" in unit:
        return dt.timedelta(days=7 * val)
    if "month" in unit:
        return dt.timedelta(days=30 * val)
    if "year" in unit:
        return dt.timedelta(days=365 * val)
    if "decade" in unit:
        return dt.timedelta(days=3650 * val)
    return None


def apply_output_guardrails(answer: str, intermediate_steps: list, station_ids: List[str], time_window: Optional[str] = None) -> OutputGuardrailResult:
    has_extract_success = False
    timestamps_present = False
    list_series_empty = False
    series_values: list[float] = []
    last_ts_ms: Optional[float] = None
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
            if tool_name in {"list_highcharts_series", "list_plotly_traces"}:
                try:
                    if isinstance(observation, dict):
                        data = observation.get("data")
                        if data == []:
                            list_series_empty = True
                except Exception:
                    pass
            continue
        if _extract_successful_series(observation):
            has_extract_success = True
            parsed_obs = observation
            if isinstance(observation, str):
                try:
                    parsed_obs = json.loads(observation)
                except Exception:
                    parsed_obs = observation
            if isinstance(parsed_obs, dict):
                data = parsed_obs.get("data") if "data" in parsed_obs else parsed_obs.get("values")
                if isinstance(data, list) and data and isinstance(data[0], dict) and "x" in data[0]:
                    timestamps_present = True
                xs = _extract_timestamps_ms(parsed_obs)
                if xs:
                    last_ts_ms = max(xs)
            series_values = _extract_series_values(observation)
            break

    if has_extract_success:
        warnings: List[str] = []
        warning_reason: Optional[str] = None
        if series_values:
            nonzero = [v for v in series_values if abs(v) > 1e-6]
            if len(nonzero) == 0 or len(nonzero) <= max(1, int(0.02 * len(series_values))):
                warnings.append(
                    "Values are at or near zero throughout this period. That can mean dry conditions, "
                    "flow below the sensor threshold, or missing/offline data. Interpret with caution."
                )
        if time_window and not timestamps_present:
            tw_note = f"Requested window ({time_window}) but the extracted series lacked timestamps; values may not align exactly."
            warnings.append(tw_note)

        if time_window and last_ts_ms is not None:
            delta = _parse_time_window_to_timedelta(time_window)
            if delta:
                last_dt = dt.datetime.utcfromtimestamp(last_ts_ms / 1000.0)
                now = dt.datetime.utcnow()
                if now - last_dt > delta:
                    warning_reason = "stale_data"
                    warnings.append(
                        f"No measurements fall within the requested window ({time_window}). "
                        f"The last available observation is from {last_dt.isoformat()}Z."
                    )

        if warnings:
            caveat = " ".join(warnings)
            adjusted = f"{answer}\n\nCaveat: {caveat}"
            return OutputGuardrailResult(triggered=False, answer=adjusted, warnings=caveat, reason=warning_reason)
        return OutputGuardrailResult(triggered=False, answer=answer)

    station = station_ids[0] if station_ids else "the requested station"
    reason = "no_chart_data"
    disclaimer = (
        f"No recent chart data was available for {station}; numeric values are omitted. "
        "Please provide another station or timeframe."
    )
    if list_series_empty:
        reason = "no_recent_data"
        disclaimer = "Charts were empty or unavailable; this station may be inactive or have no recent data. Please try another station or location."

    adjusted = disclaimer

    return OutputGuardrailResult(
        triggered=True,
        reason=reason,
        answer=adjusted,
    )

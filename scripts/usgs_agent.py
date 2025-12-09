"""
Agent definition for interactive analysis of USGS water monitoring data.

Run:
    docker compose run --rm app python scripts/usgs_agent.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import re
import numpy as np
import datetime as dt

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from openai import OpenAI
from qdrant_client.models import Filter, FieldCondition, MatchValue
from playwright.sync_api import sync_playwright
from qdrant_client import QdrantClient

from scripts.invoke_agent import invoke_agent

PW_HEADLESS = os.getenv("PW_HEADLESS", "1") != "0"
PW_ARGS = ["--disable-dev-shm-usage"]
CHART_READY_TIMEOUT_MS = int(os.getenv("CHART_READY_TIMEOUT_MS", "90000"))
NAV_TIMEOUT_MS = int(os.getenv("PW_NAV_TIMEOUT_MS", "90000"))

TMP_DIR = os.getenv("APP_TMP_DIR", "/app/.tmp")
STATION_ID_REGEX = re.compile(r"\d{7,15}")


def _get_tmp_dir() -> str:
    os.makedirs(TMP_DIR, exist_ok=True)
    return TMP_DIR


def write_last_user_query(text: str) -> None:
    """Persist the last user query for time-window inference fallbacks."""
    try:
        path = Path(_get_tmp_dir()) / "last_user_query.txt"
        path.write_text(text or "", encoding="utf-8")
    except Exception:
        # Best-effort; do not crash the agent if logging fails
        pass


def read_last_user_query() -> Optional[str]:
    """Load the most recent user query if available."""
    try:
        path = Path(_get_tmp_dir()) / "last_user_query.txt"
        if path.exists():
            return path.read_text(encoding="utf-8")
    except Exception:
        return None
    return None


def extract_station_id(text: Optional[str]) -> Optional[str]:
    """
    Extract a USGS station ID (7-15 digits) from arbitrary text.
    Returns the first valid match or None.
    """
    if not text:
        return None
    s = text.strip()
    if s.isdigit() and 7 <= len(s) <= 15:
        return s
    m = STATION_ID_REGEX.search(s)
    if not m:
        return None
    sid = m.group(0)
    if 7 <= len(sid) <= 15:
        return sid
    return None

# Infra helpers

def _get_qdrant_client():
    return QdrantClient(
        url=os.getenv("QDRANT_URL", "http://qdrant:6333"),
        prefer_grpc=False,
        timeout=float(os.getenv("QDRANT_TIMEOUT", "60"))  # type: ignore
    )


def _get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required to embed queries. Set it in your .env file."
        )
    return OpenAI(api_key=api_key, base_url=base_url)


def _wait_for_highcharts_ready(page):
    # Waits for at least one rendered Highcharts series with points or data
    page.wait_for_function(
        "() => window.Highcharts && Highcharts.charts && "
        "Highcharts.charts.filter(Boolean).some(c => c.series && c.series.length && "
        "c.series.some(s => (s.points && s.points.length) || (s.data && s.data.length)))",
        timeout=CHART_READY_TIMEOUT_MS,
    )


def _with_retry(fn, tries=2, pause_ms=800):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            if i == tries - 1:
                raise
            page = e.args[1] if len(e.args) > 1 else None  # optional
            # brief backoff + soft reload helps SPAs finish XHRs
            if page:
                page.wait_for_timeout(pause_ms * (i + 1))
                page.reload(wait_until="networkidle", timeout=NAV_TIMEOUT_MS)


# Tools

@tool
def query_metadata(
    query: str = "",
    top_k: int = 5,
    state: Optional[str] = None,  # supports "AZ", "Arizona", "AZ|CA", "Arizona, California"
    station_type: Optional[str] = None,  # supports "surface-water", "stream", "river", "ST", etc.
) -> str:
    """
    Find USGS stations by semantic query. Optional filters:
    - state: supports 'AZ', 'Arizona', 'AZ|CA', 'Arizona, California'
    - station_type: accepts 'surface-water'/'stream'/'river' -> 'surface_water', 'groundwater', 'spring'.
    Returns a JSON list with station_id, name, state, type, similarity_score.
    """
    client = _get_qdrant_client()
    openai_client = _get_openai_client()

    # Helpers
    STATE_NAME_TO_ABBR = {
        "ARIZONA": "AZ", "CALIFORNIA": "CA", "COLORADO": "CO",
        "NEW MEXICO": "NM", "UTAH": "UT", "NEVADA": "NV",
    }
    VALID = set(STATE_NAME_TO_ABBR.values())

    def norm_state_token(tok: str) -> Optional[str]:
        if not tok: return None
        t = tok.strip().upper()
        if t in VALID: return t
        return STATE_NAME_TO_ABBR.get(t, None)

    def norm_states(s: Optional[str]) -> List[str]:
        """Accept 'AZ', 'Arizona', 'AZ|CA', 'Arizona, California', etc."""
        if not s: return []
        raw = [x for part in re.split(r"[|,]", s) for x in [part.strip()] if x]
        out = []
        for r in raw:
            n = norm_state_token(r)
            if n and n not in out: out.append(n)
        return out

    def norm_station_type(st: Optional[str]) -> Optional[str]:
        if not st: return None
        t = st.lower().replace("-", " ").strip()
        if any(w in t for w in ["surface", "stream", "river", "st"]):
            return "surface_water"
        if any(w in t for w in ["ground", "gw", "well"]):
            return "groundwater"
        if "spring" in t or t == "sp":
            return "spring"
        return None  # drop unknowns
    
    effective_query = (query or "").strip()
    if not effective_query:
        parts = ["USGS station"]
        if station_type:
            parts.append(f"for {station_type}")
        if state:
            parts.append(f"in {state}")
        effective_query = " ".join(parts)

    # Embed the query
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=effective_query,
    )
    embedding = response.data[0].embedding

    # Build robust filter
    must: List[FieldCondition] = []
    should: List[FieldCondition] = []

    stype = norm_station_type(station_type)
    if stype:
        must.append(FieldCondition(key="station_type", match=MatchValue(value=stype)))

    states = norm_states(state)
    if len(states) == 1:
        must.append(FieldCondition(key="state", match=MatchValue(value=states[0])))
    elif len(states) > 1:
        for abbr in states:
            should.append(FieldCondition(key="state", match=MatchValue(value=abbr)))

    query_filter = None
    if must or should:
        query_filter = Filter(must=must or None, should=should or None)  # type: ignore

    # Query Qdrant
    results = client.query_points(
        collection_name=os.getenv("COLLECTION_NAME", "usgs_stations"),
        query=embedding,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
    )

    # Fallback: if filters returned nothing, retry without filters once
    if not results.points and (must or should):
        results = client.query_points(
            collection_name=os.getenv("COLLECTION_NAME", "usgs_stations"),
            query=embedding,
            limit=top_k,
            with_payload=True,
        )

    formatted = []
    for pt in results.points:
        payload: Dict[str, Any] = pt.payload or {}
        formatted.append({
            "station_id": payload.get("station_id"),
            "station_name": payload.get("station_name"),
            "description": payload.get("description"),
            "similarity_score": pt.score,
            "state": payload.get("state"),
            "station_type": payload.get("station_type"),
        })
    return json.dumps(formatted)



@tool
def streamstats_url_for_gage(station_id: str, tab: str = "plots") -> str:
    """
    Return the StreamStats URL for a given USGS station id.
    Example: station_id '09429500' -> https://streamstats.usgs.gov/ss/?gage=09429500&tab=plots
    """
    sid = extract_station_id(station_id)
    if not sid:
        return json.dumps({
            "success": False,
            "error": (
                f"Could not parse a valid USGS station id from {station_id!r}. "
                "Expect something like '09380000' (7-15 digits)."
            ),
        })
    url = f"https://streamstats.usgs.gov/ss/?gage={sid}&tab={tab}"
    return json.dumps({"success": True, "url": url, "station_id": sid})


@tool
def list_highcharts_series(url: str) -> str:
    """
    List Highcharts charts/series (name/type/lengths) on the page so you can pick indices or names.
    """
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=PW_HEADLESS, args=PW_ARGS)
            context = browser.new_context()
            page = context.new_page()
            page.set_default_timeout(CHART_READY_TIMEOUT_MS)
            page.set_default_navigation_timeout(NAV_TIMEOUT_MS)

            def _open():
                page.goto(url, wait_until="networkidle", timeout=NAV_TIMEOUT_MS)
                # (optional) ensure the “Plots” tab is active in case the hash isn't respected
                try:
                    page.get_by_role("tab", name="Plots", exact=True).click(timeout=2000)
                except Exception:
                    pass
                _wait_for_highcharts_ready(page)

            _with_retry(lambda: (_open(), None))
            
            js = """
            () => {
              if (!window.Highcharts || !Highcharts.charts) return [];
              const charts = Highcharts.charts.filter(Boolean);
              return charts.map((chart, ci) => {
                const series = (chart && chart.series) ? chart.series : [];
                return series.map((s, si) => ({
                  chart_index: ci,
                  series_index: si,
                  name: s && s.name ? s.name : null,
                  type: s && s.type ? s.type : null,
                  points_len: (s && (s.points || s.data)) ? (s.points || s.data).length : null
                }));
              });
            }
            """
            info = page.evaluate(js) or []

            # for debugging
            html = page.content()
            os.makedirs("/app/.tmp", exist_ok=True)
            with open("/app/.tmp/last_page.html", "w") as f:
                f.write(html)
            page.screenshot(path="/app/.tmp/last_page.png", full_page=True)

            browser.close()
            return json.dumps({"success": True, "data": info})
    except Exception as e:
        return json.dumps({"success": False, "error": f"Playwright/Highcharts failed: {e}"})


@tool
def extract_highcharts_series(
    url: str,
    chart_index: int = 0,
    series_index: int = 0,
    axis: str = "y",
    max_points: int = 2000,
    series_name_substring: Optional[str] = None,
    include_timestamps: bool = True,
) -> str:
    """
    Return one series from a Highcharts chart on the page.
    axis: 'x' or 'y'. Optionally choose the series by partial name.
    If include_timestamps is true, return objects with x (epoch ms) and y instead of plain numbers.
    """
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=PW_HEADLESS, args=PW_ARGS)
            context = browser.new_context()
            page = context.new_page()
            page.set_default_timeout(CHART_READY_TIMEOUT_MS)
            page.set_default_navigation_timeout(NAV_TIMEOUT_MS)

            def _open():
                page.goto(url, wait_until="networkidle", timeout=NAV_TIMEOUT_MS)
                # (optional) ensure the “Plots” tab is active in case the hash isn't respected
                try:
                    page.get_by_role("tab", name="Plots", exact=True).click(timeout=2000)
                except Exception:
                    pass
                _wait_for_highcharts_ready(page)

            _with_retry(lambda: (_open(), None))

            js = """
            ([ci, si, axis, nameSub, includeTs]) => {
              if (!window.Highcharts || !Highcharts.charts) {
                return {success:false, error:"Highcharts not found"};
              }
              const charts = Highcharts.charts.filter(Boolean);
              const chart = charts[ci];
              if (!chart) return {success:false, error:`No chart at index ${ci}`};

              let index = si;
              if (nameSub && typeof nameSub === 'string') {
                const sub = nameSub.toLowerCase();
                const found = chart.series.findIndex(s => (s && s.name || '').toLowerCase().includes(sub));
                if (found >= 0) index = found;
              }

              const s = chart.series[index];
              if (!s) return {success:false, error:`No series at index ${index}`};

              const pts = (s.points && s.points.length ? s.points : s.data) || [];
              const isRange = s && (s.type === 'arearange' || s.type === 'areasplinerange');
              const vals = pts.map(p => {
                const x = p && p.x;
                let val;
                if (axis === 'x') val = x;
                else if (isRange) val = (axis === 'low') ? (p && p.low) : (p && p.high);
                else val = p && p.y;
                if (includeTs && axis !== 'x') {
                  return { x, y: val };
                }
                return val;
              });
              return {success:true, values: vals};
            }
            """
            out = page.evaluate(js, [chart_index, series_index, axis, series_name_substring, include_timestamps])
            browser.close()
    except Exception as e:
        return json.dumps({"success": False, "error": f"Playwright/Highcharts failed: {e}"})

    if not out or not out.get("success"):
        return json.dumps(out or {"success": False, "error": "Unknown failure"})
    raw = out.get("values") or []
    trimmed = raw[:max_points]

    def to_float(v):
        try:
            return float(v)
        except Exception:
            return None

    if include_timestamps and trimmed and isinstance(trimmed[0], dict):
        # # DEBUG: log basic x/y range coming out of Highcharts
        # first = trimmed[0]
        # last = trimmed[-1]
        # try:
        #     first_dt = dt.datetime.utcfromtimestamp(float(first.get("x", 0)) / 1000.0)
        # except Exception:
        #     first_dt = None
        # try:
        #     last_dt = dt.datetime.utcfromtimestamp(float(last.get("x", 0)) / 1000.0)
        # except Exception:
        #     last_dt = None
        # print(
        #     "[DEBUG extract_highcharts_series] "
        #     f"n_points={len(trimmed)} "
        #     f"first_x={first.get('x')} last_x={last.get('x')} "
        #     f"first_dt={first_dt} last_dt={last_dt}",
        #     flush=True,
        # )

        cleaned = []
        for item in trimmed:
            x = item.get("x")
            y = item.get("y")
            y_val = to_float(y)
            if y_val is None:
                continue
            # normalize x to primitive timestamp if possible
            if hasattr(x, "timestamp"):
                try:
                    x = int(x.timestamp() * 1000)
                except Exception:
                    x = None
            cleaned.append({"x": x, "y": y_val})
        return json.dumps({"success": True, "data": cleaned})


    numeric = [x for x in (to_float(v) for v in trimmed) if x is not None]
    return json.dumps({"success": True, "data": numeric})


@tool
def slice_series_to_time_window(
    series: Any,
    window: str = "",
) -> str:
    """
    Slice a time series to a requested window. If window is missing/blank, this tool
    will attempt to infer a window or absolute date range from the last user query
    persisted in /app/.tmp/last_user_query.txt.

    `series` can be:
      - Raw JSON string from extract_highcharts_series or extract_plotly_series
      - Dict with `data` or `values` containing [{x, y}]
      - Plain list of {x, y}

    `window` may be:
      - Relative: '24 hours', '48 hours', '7 days', '2 months', '1 year',
        'last week', 'last month', 'last year', 'last decade'
      - Omitted: we try to infer from the last user query
      - Absolute ranges in the last query (e.g., "from 2025-01-05 to 2025-01-21")

    Returns JSON with:
      success: true/false
      data: sliced points
      window_used: canonical window or 'all_data'
      last_timestamp_utc: ISO8601 of the most recent point in the full series
      count: number of points in the returned data
      error: present when success is false
    """
    # # DEBUG: log what the tool actually got
    # print(
    #     f"[DEBUG slice_series_to_time_window] called with window={window!r}, type(series)={type(series).__name__}",
    #     flush=True,
    # )

    # --- Helper: coerce x into epoch ms (float) ---
    def _coerce_ms(x_val_raw: Any) -> Optional[float]:
        # Already numeric
        if isinstance(x_val_raw, (int, float)):
            try:
                return float(x_val_raw)
            except Exception:
                return None

        # datetime-like object
        if hasattr(x_val_raw, "timestamp"):
            try:
                return float(x_val_raw.timestamp() * 1000.0)
            except Exception:
                return None

        # String: could be numeric ms or ISO-ish datetime
        if isinstance(x_val_raw, str):
            s = x_val_raw.strip()
            if not s:
                return None

            # Numeric string (e.g., "1732924800000")
            try:
                return float(s)
            except Exception:
                pass

            # ISO-ish datetime (e.g., "2025-11-23 12:15:00+00:00" or "...Z")
            try:
                iso = s.replace("Z", "+00:00")
                # Normalize space to 'T' if needed
                if " " in iso and "T" not in iso:
                    iso = iso.replace(" ", "T")
                dt_obj = dt.datetime.fromisoformat(iso)
                return float(dt_obj.timestamp() * 1000.0)
            except Exception:
                return None

        return None

    def _parse_absolute_range_from_last_query() -> tuple[Optional[float], Optional[float], Optional[str]]:
        """
        Read the last user query, extract ISO-like datetime tokens, and return (start_ms, end_ms, label).
        Tokens supported: YYYY-MM-DD with optional time and optional Z / offset.
        """
        try:
            query_text = read_last_user_query()
        except Exception:
            query_text = None
        if not query_text:
            return None, None, None
        pattern = re.compile(
            r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?(?:Z|[+-]\d{2}:\d{2})?"
        )
        times_ms: List[float] = []
        for match in pattern.findall(query_text):
            ms = _coerce_ms(match)
            if ms is not None:
                times_ms.append(ms)
        if len(times_ms) < 2:
            return None, None, None
        start_ms = min(times_ms)
        end_ms = max(times_ms)
        start_iso = dt.datetime.utcfromtimestamp(start_ms / 1000.0).isoformat() + "Z"
        end_iso = dt.datetime.utcfromtimestamp(end_ms / 1000.0).isoformat() + "Z"
        return start_ms, end_ms, f"{start_iso} to {end_iso}"

    def _infer_relative_window_from_query(query_text: str) -> Optional[str]:
        """
        Try to infer a relative window string from the last user query.
        Returns a canonical string like '24 hours' or '7 days', or None.
        """
        if not query_text:
            return None
        txt = query_text.lower()
        # explicit keywords
        specials = {
            "last week": "7 days",
            "last month": "30 days",
            "last year": "365 days",
            "last decade": "10 years",
        }
        for key, val in specials.items():
            if key in txt:
                return val
        m = re.search(r"last\s+(\d+)\s*(hour|hours|day|days|week|weeks|month|months|year|years|decade|decades)", txt)
        if m:
            val = int(m.group(1))
            unit = m.group(2)
            if "hour" in unit:
                return f"{val} hours"
            if "day" in unit:
                return f"{val} days"
            if "week" in unit:
                return f"{val} weeks"
            if "month" in unit:
                return f"{val} months"
            if "year" in unit:
                return f"{val} years"
            if "decade" in unit:
                return f"{val} decades"
        return None

    def _parse_window_delta(win_str: str) -> tuple[Optional[dt.timedelta], Optional[str]]:
        win = (win_str or "").strip().lower()
        delta: Optional[dt.timedelta] = None
        canonical: Optional[str] = None
        if not win:
            return None, None
        if win in {"last week"}:
            delta = dt.timedelta(days=7)
            canonical = "7 days"
        elif win in {"last month"}:
            delta = dt.timedelta(days=30)
            canonical = "30 days"
        elif win in {"last year"}:
            delta = dt.timedelta(days=365)
            canonical = "365 days"
        elif win in {"last decade"}:
            delta = dt.timedelta(days=3650)
            canonical = "10 years"
        else:
            m = re.search(
                r"(\d+)\s*(hour|hours|day|days|week|weeks|month|months|year|years|decade|decades)",
                win,
            )
            if m:
                val = int(m.group(1))
                unit = m.group(2)
                if "hour" in unit:
                    delta = dt.timedelta(hours=val)
                    canonical = f"{val} hours"
                elif "day" in unit:
                    delta = dt.timedelta(days=val)
                    canonical = f"{val} days"
                elif "week" in unit:
                    delta = dt.timedelta(days=7 * val)
                    canonical = f"{val} weeks"
                elif "month" in unit:
                    delta = dt.timedelta(days=30 * val)
                    canonical = f"{val} months"
                elif "year" in unit:
                    delta = dt.timedelta(days=365 * val)
                    canonical = f"{val} years"
                elif "decade" in unit:
                    delta = dt.timedelta(days=3650 * val)
                    canonical = f"{val} decades"
        return delta, canonical

    # --- Step 1: normalize `series` into a raw list of points ---
    points_raw: Optional[List[Any]] = None

    if series is None:
        return json.dumps({
            "success": False,
            "error": "Missing 'series' argument; pass the JSON from extract_* as 'series'.",
        })

    # If model passed the raw JSON string
    if isinstance(series, str):
        try:
            parsed = json.loads(series)
        except Exception:
            parsed = None

        if isinstance(parsed, dict):
            maybe = parsed.get("data") or parsed.get("values")
            if isinstance(maybe, list):
                points_raw = maybe
        elif isinstance(parsed, list):
            points_raw = parsed

    # If model passed a dict (already-parsed JSON)
    elif isinstance(series, dict):
        maybe = series.get("data") or series.get("values")
        if isinstance(maybe, list):
            points_raw = maybe

    # If model passed a list directly
    elif isinstance(series, list):
        points_raw = series

    if not isinstance(points_raw, list) or not points_raw:
        return json.dumps({
            "success": False,
            "error": "Could not extract a list of points from 'series'; "
                     "expected something like {'data': [{'x': ..., 'y': ...}, ...]}.",
        })

    # --- Step 2: clean into [{x, y}] with x as epoch ms (float), y as float ---
    xs: List[float] = []
    cleaned: List[dict] = []
    for p in points_raw:
        if not isinstance(p, dict):
            continue
        x_raw = p.get("x")
        y_raw = p.get("y")
        if x_raw is None or y_raw is None:
            continue

        x_ms = _coerce_ms(x_raw)
        if x_ms is None:
            continue

        try:
            y_val = float(y_raw)
        except Exception:
            continue

        xs.append(x_ms)
        cleaned.append({"x": x_ms, "y": y_val})

    if not xs:
        return json.dumps({"success": False, "error": "No valid timestamps in series"})

    last_ms = max(xs)
    last_dt = dt.datetime.utcfromtimestamp(last_ms / 1000.0)

    win_str = (window or "").strip()
    absolute_range = None
    last_query_text = None
    if not win_str:
        try:
            last_query_text = read_last_user_query()
        except Exception:
            last_query_text = None
        absolute_range = _parse_absolute_range_from_last_query()
        if absolute_range and (absolute_range[0] is None or absolute_range[1] is None):
            absolute_range = None
        if not absolute_range and last_query_text:
            inferred = _infer_relative_window_from_query(last_query_text)
            if inferred:
                win_str = inferred

    # Absolute date range slicing (if inferred)
    if absolute_range:
        start_ms, end_ms, label = absolute_range
        if start_ms is not None and end_ms is not None:
            if start_ms > end_ms:
                start_ms, end_ms = end_ms, start_ms
            sliced = [p for p in cleaned if start_ms <= p["x"] <= end_ms]
            print(
                "[DEBUG slice_series_to_time_window] absolute_range "
                f"start_ms={start_ms} end_ms={end_ms} "
                f"n_points={len(cleaned)} sliced_points={len(sliced)} "
                f"last_dt={last_dt.isoformat()}Z",
                flush=True,
            )
            return json.dumps({
                "success": True,
                "data": sliced,
                "window_used": label or "absolute_range",
                "last_timestamp_utc": last_dt.isoformat() + "Z",
                "count": len(sliced),
            })

    if not win_str:
        print(
            "[DEBUG slice_series_to_time_window] no_valid_window "
            f"window={window!r} n_points={len(cleaned)} last_dt={last_dt.isoformat()}Z",
            flush=True,
        )
        return json.dumps({
            "success": False,
            "data": cleaned,
            "window_used": "all_data",
            "last_timestamp_utc": last_dt.isoformat() + "Z",
            "count": len(cleaned),
            "error": "Could not determine a valid time window from 'window' or the last user query.",
        })

    delta, canonical = _parse_window_delta(win_str)

    # If we still couldn't parse the window, return all data but log it
    if not delta:
        print(
            "[DEBUG slice_series_to_time_window] parse_failed "
            f"window={win_str!r} win_norm={win_str.lower()!r} "
            f"n_points={len(cleaned)} last_dt={last_dt.isoformat()}Z",
            flush=True,
        )
        return json.dumps({
            "success": False,
            "data": cleaned,
            "window_used": "all_data",
            "last_timestamp_utc": last_dt.isoformat() + "Z",
            "count": len(cleaned),
            "error": f"Could not parse window={win_str!r}",
        })

    cutoff_dt = last_dt - delta
    cutoff_ms = cutoff_dt.timestamp() * 1000.0
    sliced = [p for p in cleaned if p["x"] >= cutoff_ms]

    # --- DEBUG: log slice behavior ---
    first_full = cleaned[0]
    last_full = cleaned[-1]
    first_slice = sliced[0] if sliced else None
    last_slice = sliced[-1] if sliced else None

    def _fmt_ms(ms: Optional[float]) -> Optional[str]:
        try:
            return dt.datetime.utcfromtimestamp(float(ms) / 1000.0).isoformat() + "Z"
        except Exception:
            return None

    print(
        "[DEBUG slice_series_to_time_window] "
        f"window={window!r} canonical={canonical!r} "
        f"total_points={len(cleaned)} sliced_points={len(sliced)} "
        f"last_dt={last_dt.isoformat()}Z "
        f"full_first_dt={_fmt_ms(first_full.get('x'))} "
        f"full_last_dt={_fmt_ms(last_full.get('x'))} "
        f"sliced_first_dt={_fmt_ms(first_slice.get('x')) if first_slice else None} "
        f"sliced_last_dt={_fmt_ms(last_slice.get('x')) if last_slice else None}",
        flush=True,
    )

    return json.dumps({
        "success": True,
        "data": sliced,
        "window_used": canonical or win_str,
        "last_timestamp_utc": last_dt.isoformat() + "Z",
        "count": len(sliced),
    })


@tool
def list_plotly_traces(url: str) -> str:
    """List charts and traces (name/type/lengths) so you can pick indices."""
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=PW_HEADLESS, args=PW_ARGS)
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded")
            page.wait_for_selector(".plotly, .plotly-graph-div, .js-plotly-plot", timeout=15000)
            js = """
            () => {
              const sels = ['.plotly','.plotly-graph-div','.js-plotly-plot'];
              const charts = document.querySelectorAll(sels.join(','));
              return Array.from(charts).map((div, ci) => {
                const data = div.data || [];
                return data.map((tr, ti) => ({
                  chart_index: ci,
                  trace_index: ti,
                  name: tr.name ?? null,
                  type: tr.type ?? null,
                  x_len: Array.isArray(tr.x) ? tr.x.length : null,
                  y_len: Array.isArray(tr.y) ? tr.y.length : null,
                  has_z: !!tr.z
                }));
              });
            }"""
            info = page.evaluate(js) or []
            browser.close()
            return json.dumps({ "success": True, "data": info })
    except Exception as e:
        return json.dumps({ "success": False, "error": f"Playwright failed: {e}" })


@tool
def extract_plotly_series(
    url: str,
    chart_index: int = 0,
    trace_index: int = 0,
    axis: str = "y",
    max_points: int = 2000,
    trace_name_substring: Optional[str] = None,
    include_timestamps: bool = False,
) -> str:
    """
    Return ONLY one series from a Plotly chart on the page.
    axis: 'x' | 'y' | 'z'. Optionally choose the trace by name substring.
    If include_timestamps is true and axis is y, include paired x values.
    """
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=PW_HEADLESS, args=PW_ARGS)
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded")
            page.wait_for_selector(".plotly, .plotly-graph-div, .js-plotly-plot", timeout=15000)

            js = """
            ([ci, ti, axis, nameSub]) => {
              const sels = ['.plotly','.plotly-graph-div','.js-plotly-plot'];
              const charts = document.querySelectorAll(sels.join(','));
              const div = charts[ci];
              if (!div) return {success:false,error:`No chart at index ${ci}`};

              const data = div.data || [];
              let index = ti;
              if (nameSub && typeof nameSub === 'string') {
                const sub = nameSub.toLowerCase();
                const found = data.findIndex(tr => (tr.name ?? '').toLowerCase().includes(sub));
                if (found >= 0) index = found;
              }

              const tr = data[index];
              if (!tr) return {success:false,error:`No trace at index ${index}`};

              let arr = (axis==='x') ? tr.x : (axis==='z') ? tr.z : tr.y;
              if (!arr) return {success:false,error:`Trace has no axis ${axis}`};

              return {success:true, values: Array.isArray(arr) ? arr : []};
            }
            """
            out = page.evaluate(js, [chart_index, trace_index, axis, trace_name_substring])
            browser.close()
    except Exception as e:
        return json.dumps({"success": False, "error": f"Playwright failed: {e}"})

    if not out or not out.get("success"):
        return json.dumps(out or {"success": False, "error": "Unknown failure"})
    vals = out["values"] or []

    # Trim and coerce to float where possible
    trimmed = vals[:max_points]
    def to_float(x):
        try:
            if isinstance(x, dict) and "v" in x:
                x = x["v"]
            return float(x)  # type: ignore
        except:
            return None

    if include_timestamps and axis == "y" and out.get("values") and "x" in (out.get("values")[0] if isinstance(out.get("values")[0], dict) else {}):
        paired = []
        xs = out.get("values_x") or []
        for item in trimmed:
            if isinstance(item, dict) and "x" in item and "y" in item:
                y_val = to_float(item["y"])
                if y_val is None:
                    continue
                paired.append({"x": item["x"], "y": y_val})
        if paired:
            return json.dumps({"success": True, "data": paired})

    numeric = [to_float(v) for v in trimmed]
    numeric = [v for v in numeric if v is not None]
    return json.dumps({"success": True, "data": numeric})


@tool
def calculate_statistics(values: List[float]) -> str:
    """
    Compute count, mean, median, std, min, and max for a list of numeric values.

    When you are working with time-series points of the form
    objects that have fields x (timestamp in ms) and y (value),
    you MUST pass only the numeric y values here.

    Example usage:
      - First call extract_highcharts_series(..., include_timestamps=True)
        to get a JSON result whose 'data' field is a list of points with x and y.
      - Optionally call slice_series_to_time_window(series=<that JSON result>, window='7 days')
        to get a sliced subset.
      - Then call calculate_statistics with a plain list of y values from
        either the full series or the sliced subset, e.g.:
        values = [p['y'] for p in sliced_data]
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return json.dumps({"success": False, "error": "Empty dataset"})

    # # DEBUG: log what we're actually summarizing
    # print(
    #     "[DEBUG calculate_statistics] "
    #     f"n={arr.size} min={float(arr.min())} max={float(arr.max())}",
    #     flush=True,
    # )

    stats = {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std_dev": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
    return json.dumps({"success": True, "data": stats})


@tool
def calculate_growth_rate(values: List[float]) -> str:
    """Compute period-over-period growth rates and their average."""
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return json.dumps({"success": False, "error": "Need at least two values"})
    growth_rates: List[Optional[float]] = []
    for i in range(1, arr.size):
        prev = arr[i - 1]
        current = arr[i]
        growth_rates.append(None if prev == 0 else (current - prev) / prev)
    non_null = [x for x in growth_rates if x is not None]
    avg_growth = float(np.mean(non_null)) if non_null else None
    return json.dumps({"success": True, "data": {"growth_rates": growth_rates, "average_growth_rate": avg_growth}})


@tool
def detect_outliers(values: List[float], method: str = "iqr", iqr_multiplier: float = 1.5) -> str:
    """Identify outliers via IQR."""
    arr = np.asarray(values, dtype=float)
    if arr.size < 3:
        return json.dumps({"success": False, "error": "Need at least three values"})
    if method != "iqr":
        return json.dumps({"success": False, "error": f"Unknown method '{method}'"})
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    lb = q1 - iqr_multiplier * iqr
    ub = q3 + iqr_multiplier * iqr
    outliers = [float(x) for x in arr[(arr < lb) | (arr > ub)]]
    return json.dumps({"success": True, "data": {"outliers": outliers, "lower_bound": float(lb), "upper_bound": float(ub)}})


# Agent builder

def build_agent_executor() -> AgentExecutor:
    tools_list = [
        query_metadata,
        streamstats_url_for_gage,
        list_highcharts_series,
        extract_highcharts_series,
        list_plotly_traces,
        extract_plotly_series,
        slice_series_to_time_window,
        calculate_statistics,
        calculate_growth_rate,
        detect_outliers,
    ]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You analyze public USGS charts.\n"
                "\n"
                "GENERAL BEHAVIOR\n"
                "- Always base numeric answers on values that come from tools.\n"
                "- Provide the time window string when calling slice_series_to_time_window; the tool can infer from the last user query if omitted, but you should still send an explicit window.\n"
                "- Never invent time ranges or statistics; if a requested time window has no data, say so explicitly.\n"
                "- Assume units are streamflow/discharge in cubic feet per second (cfs) unless stated otherwise.\n"
                "\n"
                "WHEN THE USER DOES NOT GIVE A STATION ID\n"
                "- First call query_metadata to find candidate stations.\n"
                "  - Normalize state filters (for example AZ|CA or 'Arizona, California').\n"
                "  - Prefer station_type='surface_water' when appropriate.\n"
                "  - If a filtered search returns nothing, retry query_metadata without filters and pick the best match by name/location.\n"
                "\n"
                "WHEN THE USER DOES GIVE A STATION ID\n"
                "- Call streamstats_url_for_gage to get the StreamStats URL for that station.\n"
                "- Optionally call query_metadata to get extra context (name/state/type), but do NOT assume the station is invalid just because metadata search returns nothing.\n"
                "- On the StreamStats URL, try list_highcharts_series first. If that returns no useful series or fails, try list_plotly_traces instead.\n"
                "- When choosing a series for 'current' or 'last N hours/days/weeks/months/years' questions:\n"
                "  - Prefer series that represent instantaneous or daily streamflow (for example names like 'Instantaneous Streamflow' or 'Discharge, cubic feet per second').\n"
                "  - Avoid annual summaries or percentile curves when the user is asking about recent or current conditions.\n"
                "\n"
                "EXTRACTING DATA FROM CHARTS\n"
                "- Once you decide which series to use, call the matching extractor:\n"
                "  - extract_highcharts_series for Highcharts charts.\n"
                "  - extract_plotly_series for Plotly charts.\n"
                "- For questions involving any time window (for example 'last 24 hours', 'last 48 hours', 'last 7 days', 'last 2 months', 'last year', 'last decade'):\n"
                "  - Call the extractor with include_timestamps=True so the result includes time stamps.\n"
                "  - The extractor returns a JSON result whose data field is a list of points with fields x (Unix time in milliseconds) and y (value).\n"
                "\n"
                "SLICING TO A TIME WINDOW\n"
                "- When the user mentions a specific recent time window, such as:\n"
                "    'last 24 hours', 'last 48 hours', 'last 7 days', 'last 2 months',\n"
                "    'last week', 'last month', 'last year', 'last decade',\n"
                "  and you have a time series with x timestamps and y values:\n"
                "  1) Take the full JSON output returned by the extractor.\n"
                "  2) Call slice_series_to_time_window with:\n"
                "       - series = that JSON result from the extractor (or an equivalent dict with a data list of points).\n"
                "       - window = a concise time window string, usually the same phrase the user used (for example 'last 2 months' or '24 hours').\n"
                "     - If you omit the window or pass an empty string, the tool will infer it from the user's question, including absolute ranges like 'between 2025-11-21 and 2025-11-23' or relative phrases like 'last 24 hours'.\n"
                "     - Prefer to provide the window explicitly when it is clear, but omission will not crash the call.\n"
                "     - If the user says 'last week', call with window='last week'. If they say 'last 24 hours', call with window='24 hours'.\n"
                "  3) If slice_series_to_time_window returns success equal to true and count greater than zero:\n"
                "       - Use ONLY the sliced subset in its data field for further calculations.\n"
                "       - For statistics, build a plain list of y values from those sliced points and pass that list into calculate_statistics.\n"
                "  4) If slice_series_to_time_window returns success equal to false or count equal to zero:\n"
                "       - Do NOT pretend there is data in that time window.\n"
                "       - Say clearly that there are no measurements in the requested window (even if the tool inferred the window).\n"
                "       - Report the date of the last available observation instead, using last_timestamp_utc or the last point's time.\n"
                "\n"
                "ERROR HANDLING FOR slice_series_to_time_window\n"
                "- If slice_series_to_time_window returns success=false (including missing/blank window):\n"
                "  - Treat that as no valid sliced data for the requested window.\n"
                "  - Do NOT fabricate numbers; instead, report that no measurements fall in that window and mention the last observation date provided.\n"
                "  - Only ask the user to clarify if the time period is genuinely ambiguous.\n"
                "RUNNING MATH AND ANALYSIS TOOLS\n"
                "- Use calculate_statistics when the user wants summaries like mean, median, minimum, maximum, standard deviation, or a count of points.\n"
                "- Use calculate_growth_rate when the user asks which station is rising or falling faster.\n"
                "- Use detect_outliers when the user asks about unusual spikes or outliers.\n"
                "- In all cases where you start from a list of points with x and y:\n"
                "  - Build a list of y values only, and pass that list as the values argument to the math tool.\n"
                "  - Never pass x timestamps into the math tools.\n"
                "\n"
                "HOW TO ANSWER\n"
                "- Return concise numeric answers, clearly tied to the station, time window, and metric.\n"
                "- Always echo the requested time window if there was one.\n"
                "- Always mention the date of the last observation you actually used in your reasoning.\n"
                "- If charts are empty, missing, or clearly not time-series data for the question, say so and suggest another station or timeframe.\n"
                "- When there is no data in the requested window, say that explicitly and do NOT fabricate statistics.\n"
                "\n"
                "EXAMPLE PLAN (ILLUSTRATIVE)\n"
                "User: 'Find a surface-water station near Imperial Dam (AZ/CA) and report the mean Instantaneous Streamflow for the last 7 days.'\n"
                "Plan:\n"
                "  1) Call query_metadata with a query such as 'Imperial Dam Colorado River', state='AZ|CA', and station_type='surface-water'.\n"
                "  2) Pick the best match and call streamstats_url_for_gage with that station_id.\n"
                "  3) Call list_highcharts_series on the resulting URL to pick a relevant time-series streamflow series.\n"
                "  4) Call extract_highcharts_series with that URL and series_name_substring='Instantaneous Streamflow', axis='y', include_timestamps=True.\n"
                "  5) Call slice_series_to_time_window with series set to the JSON result from step 4 and window='7 days'.\n"
                "  6) From the sliced data, build a list of y values and call calculate_statistics with that list.\n"
                "  7) Answer with the mean, minimum, maximum, and last observation date, explicitly referring to the last 7 days.\n"
            ),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        temperature=0,
        max_tokens=500,
    )

    agent = create_tool_calling_agent(llm, tools_list, prompt)  # type: ignore
    executor = AgentExecutor(agent=agent, tools=tools_list, verbose=False)  # type: ignore
    return executor


# CLI

def main() -> None:
    print("USGS Analysis Agent (LangChain 0.2.x / Tool-calling)\nPress Ctrl+C to exit.")

    while True:
        try:
            query = input(">>> ").strip()
            if not query:
                continue
            write_last_user_query(query)
            result = invoke_agent(query, metadata={"source": "cli"})
            print(result.get("output", ""))

        except (KeyboardInterrupt, EOFError):
            print("\nExiting agent.")
            break


if __name__ == "__main__":
    main()

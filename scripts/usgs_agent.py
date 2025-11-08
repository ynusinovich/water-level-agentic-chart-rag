"""
Agent definition for interactive analysis of USGS water monitoring data.

Run:
    docker compose run --rm app python scripts/usgs_agent.py
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional
import re
import numpy as np

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from openai import OpenAI
from qdrant_client.models import Filter, FieldCondition, MatchValue
from playwright.sync_api import sync_playwright

PW_HEADLESS = os.getenv("PW_HEADLESS", "1") != "0"
PW_ARGS = ["--disable-dev-shm-usage"]
CHART_READY_TIMEOUT_MS = int(os.getenv("CHART_READY_TIMEOUT_MS", "60000"))
NAV_TIMEOUT_MS = int(os.getenv("PW_NAV_TIMEOUT_MS", "60000"))


# Infra helpers

def _get_qdrant_client():
    from qdrant_client import QdrantClient
    import os
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
    query: str,
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

    # Embed the query
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=query,
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
    sid = (station_id or "").strip()
    if not sid.isdigit():
        return json.dumps({"success": False, "error": "station_id must be digits only"})
    url = f"https://streamstats.usgs.gov/ss/?gage={sid}&tab={tab}"
    return json.dumps({"success": True, "url": url})


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

            # # for debugging
            # html = page.content()
            # import os
            # os.makedirs("/app/.tmp", exist_ok=True)
            # with open("/app/.tmp/last_page.html", "w") as f:
            #     f.write(html)
            # page.screenshot(path="/app/.tmp/last_page.png", full_page=True)

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
) -> str:
    """
    Return one numeric series from a Highcharts chart on the page.
    axis: 'x' or 'y'. Optionally choose the series by partial name.
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
            ([ci, si, axis, nameSub]) => {
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
                if (axis === 'x') return p && p.x;
                if (isRange) return (axis === 'low') ? (p && p.low) : (p && p.high);
                return p && p.y;
              });
              return {success:true, values: vals};
            }
            """
            out = page.evaluate(js, [chart_index, series_index, axis, series_name_substring])
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

    numeric = [x for x in (to_float(v) for v in trimmed) if x is not None]
    return json.dumps({"success": True, "data": numeric})


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
) -> str:
    """
    Return ONLY one numeric series from a Plotly chart on the page.
    axis: 'x' | 'y' | 'z'. Optionally choose the trace by name substring.
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
            # Handle objects like {"x":..., "y":...} just in case
            if isinstance(x, dict) and "v" in x:
                x = x["v"]
            return float(x)  # type: ignore
        except:
            return None
    numeric = [to_float(v) for v in trimmed]
    numeric = [v for v in numeric if v is not None]
    return json.dumps({"success": True, "data": numeric})


@tool
def calculate_statistics(values: List[float]) -> str:
    """Compute count, mean, median, std, min, max for a list of numbers."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return json.dumps({"success": False, "error": "Empty dataset"})
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
        calculate_statistics,
        calculate_growth_rate,
        detect_outliers,
    ]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
            "You analyze public USGS charts.\n"
            "If the user does NOT give a USGS station_id:\n"
            "  • First call `query_metadata` to find candidate stations. Normalize states (AZ|CA etc.) and prefer station_type='surface_water'. "
            "    If a filtered search returns nothing, retry `query_metadata` without filters, then pick the best match by name/location.\n"
            "If you have a station_id:\n"
            "  • Call `streamstats_url_for_gage` to get the StreamStats URL.\n"
            "  • On that URL, try `list_highcharts_series` first; if empty or fails, try `list_plotly_traces`.\n"
            "  • Choose the correct series by name (e.g., 'Annual Peak Streamflow', 'Daily Percentile Streamflow'); "
            "    then call the matching extractor (`extract_highcharts_series` or `extract_plotly_series`).\n"
            "  • Run the math tool you need (`calculate_statistics`, `calculate_growth_rate`, or `detect_outliers`).\n"
            "Example:\n"
            "User: 'Find a surface-water station near Imperial Dam (AZ/CA) and report the mean Annual Peak Streamflow.'\n"
            "Plan:\n"
            "  1) query_metadata(query='Imperial Dam Colorado River', state='AZ|CA', station_type='surface-water')\n"
            "  2) streamstats_url_for_gage(station_id=<best match>)\n"
            "  3) list_highcharts_series(url)\n"
            "  4) extract_highcharts_series(url, series_name_substring='Annual Peak Streamflow', axis='y')\n"
            "  5) calculate_statistics(values)\n"
            "Return concise numeric answers. If charts aren’t found, say so and ask for another URL. "
            "Avoid water-quality pages; focus on StreamStats hydrographs for flow/peak-flow."),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
        max_tokens=500
    )

    agent = create_tool_calling_agent(llm, tools_list, prompt)  # type: ignore
    executor = AgentExecutor(agent=agent, tools=tools_list, verbose=False)  # type: ignore
    return executor


# CLI

def main() -> None:
    executor = build_agent_executor()
    print("USGS Analysis Agent (LangChain 0.2.x / Tool-calling)\nPress Ctrl+C to exit.")

    while True:
        try:
            query = input(">>> ").strip()
            if not query:
                continue

            result = executor.invoke(
                {"input": query},
                config={"tags": ["dev", "usgs"], "metadata": {"source": "cli"}},
            )
            answer = result.get("output", result)
            print(answer)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting agent.")
            break


if __name__ == "__main__":
    main()

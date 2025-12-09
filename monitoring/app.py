from __future__ import annotations

import os
import json
from typing import Optional

import streamlit as st

from monitoring.db import Database


def _safe_json(text: Optional[str]):
    if not text:
        return []
    try:
        return json.loads(text)
    except Exception:
        return text


def main():
    st.set_page_config(page_title="USGS Agent Monitor", layout="wide")
    st.title("USGS Agent Monitor")
    st.caption("Logged runs, guardrails, and evaluation checks")

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        st.error("DATABASE_URL is required.")
        return

    db = Database(db_url)
    db.ensure_schema()

    with st.sidebar:
        st.subheader("Filters")
        question_type = st.text_input(
            "Question type contains (optional)", 
            ""
        )
        guardrail = st.selectbox("Guardrail triggered", options=["Any", "Yes", "No"], index=0)
        page_size = st.number_input("Page size", min_value=10, max_value=200, value=50, step=10)

    guardrail_filter = None
    if guardrail == "Yes":
        guardrail_filter = True
    elif guardrail == "No":
        guardrail_filter = False

    logs = db.list_logs(limit=int(page_size), question_type=question_type or None, guardrail=guardrail_filter)

    st.subheader("Recent runs")
    if not logs:
        st.info("No logs found.")
        return

    options = [f"#{row['id']} • {row['created_at']} • station {row.get('station_id') or '?'}" for row in logs]
    selected = st.selectbox("Select a log", options)
    selected_id = logs[options.index(selected)]["id"]

    log = db.get_log(selected_id)
    checks = db.get_checks(selected_id)
    feedbacks = db.get_feedback(selected_id)

    st.markdown("**Overview**")
    col1, col2, col3 = st.columns(3)
    col1.write(f"Station: {log.get('station_id') or '-'}")
    col1.write(f"Question type: {log.get('question_type') or '-'}")
    col2.write(f"Guardrail: {'Yes' if log.get('guardrail_triggered') else 'No'}")
    col2.write(f"Reason: {log.get('guardrail_reason') or '-'}")
    col3.write(f"Tokens (in/out): {(log.get('total_input_tokens') or 0)} / {(log.get('total_output_tokens') or 0)}")

    with st.expander("Prompt", expanded=False):
        st.code(log.get("user_prompt") or "")
    with st.expander("Answer", expanded=True):
        st.write(log.get("assistant_answer") or "")
    with st.expander("Tool calls", expanded=False):
        st.json(_safe_json(log.get("tools_json")))

    st.markdown("**Evaluation Checks**")
    if checks:
        st.dataframe(checks, use_container_width=True)
    else:
        st.info("No checks for this log.")

    st.markdown("**Feedback**")
    if feedbacks:
        st.dataframe(feedbacks, use_container_width=True)
    else:
        st.info("No feedback yet.")

    st.write("Add feedback:")
    with st.form("feedback_form", clear_on_submit=True):
        is_good = st.radio("Is the answer good?", options=["👍 Yes", "👎 No"], horizontal=True)
        comments = st.text_area("Comments", placeholder="Notes")
        ref = st.text_area("Reference answer (optional)")
        submit = st.form_submit_button("Submit")
        if submit:
            try:
                from monitoring.db import Feedback

                fb = Feedback(
                    log_id=selected_id,
                    is_good=is_good.startswith("👍"),
                    comments=comments or None,
                    reference_answer=ref or None,
                )
                db.insert_feedback(fb)
                st.success("Feedback saved.")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Failed to save feedback: {e}")


if __name__ == "__main__":
    main()

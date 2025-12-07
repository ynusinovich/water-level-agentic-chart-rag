from __future__ import annotations

import os
import psycopg
from contextlib import contextmanager
from typing import Iterable, Optional

from .schemas import LLMLogRecord, CheckResult, Feedback


class Database:
    def __init__(self, database_url: Optional[str] = None) -> None:
        self.database_url = database_url or os.environ.get("DATABASE_URL")
        if not self.database_url:
            raise RuntimeError("DATABASE_URL is required for monitoring database (Postgres only).")
        self._conn = None

    def connect(self):
        if self._conn:
            return self._conn
        self._conn = psycopg.connect(self.database_url, autocommit=True)
        return self._conn

    @contextmanager
    def cursor(self):
        conn = self.connect()
        cur = conn.cursor()
        try:
            yield cur
        finally:
            cur.close()

    def ensure_schema(self) -> None:
        with self.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_logs (
                    id SERIAL PRIMARY KEY,
                    filepath TEXT NOT NULL,
                    agent_name TEXT,
                    provider TEXT,
                    model TEXT,
                    user_prompt TEXT,
                    station_id TEXT,
                    state_filter TEXT,
                    time_window TEXT,
                    question_type TEXT,
                    guardrail_triggered BOOLEAN,
                    guardrail_reason TEXT,
                    total_input_tokens BIGINT,
                    total_output_tokens BIGINT,
                    assistant_answer TEXT,
                    tools_json TEXT,
                    raw_json TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS eval_checks (
                    id SERIAL PRIMARY KEY,
                    log_id INTEGER NOT NULL REFERENCES llm_logs(id) ON DELETE CASCADE,
                    check_name TEXT NOT NULL,
                    passed BOOLEAN,
                    score DOUBLE PRECISION,
                    details TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    id SERIAL PRIMARY KEY,
                    log_id INTEGER NOT NULL REFERENCES llm_logs(id) ON DELETE CASCADE,
                    is_good BOOLEAN NOT NULL,
                    comments TEXT,
                    reference_answer TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                """
            )

    def insert_log(self, rec: LLMLogRecord) -> int:
        with self.cursor() as cur:
            cur.execute(
                """
                INSERT INTO llm_logs (
                    filepath, agent_name, provider, model, user_prompt,
                    station_id, state_filter, time_window, question_type,
                    guardrail_triggered, guardrail_reason,
                    total_input_tokens, total_output_tokens,
                    assistant_answer, tools_json, raw_json
                )
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                RETURNING id;
                """,
                (
                    rec.filepath,
                    rec.agent_name,
                    rec.provider,
                    rec.model,
                    rec.user_prompt,
                    rec.station_id,
                    rec.state_filter,
                    rec.time_window,
                    rec.question_type,
                    rec.guardrail_triggered,
                    rec.guardrail_reason,
                    rec.total_input_tokens,
                    rec.total_output_tokens,
                    rec.assistant_answer,
                    rec.tools_json,
                    rec.raw_json,
                ),
            )
            return int(cur.fetchone()[0])

    def insert_checks(self, checks: Iterable[CheckResult]) -> None:
        checks = list(checks)
        if not checks:
            return
        with self.cursor() as cur:
            for c in checks:
                cur.execute(
                    """
                    INSERT INTO eval_checks (log_id, check_name, passed, score, details)
                    VALUES (%s,%s,%s,%s,%s);
                    """,
                    (
                        c.log_id,
                        getattr(c.check_name, "value", str(c.check_name)),
                        c.passed,
                        c.score,
                        c.details,
                    ),
                )

    def insert_feedback(self, fb: Feedback) -> int:
        with self.cursor() as cur:
            cur.execute(
                """
                INSERT INTO feedback (log_id, is_good, comments, reference_answer)
                VALUES (%s,%s,%s,%s)
                RETURNING id;
                """,
                (fb.log_id, fb.is_good, fb.comments, fb.reference_answer),
            )
            return int(cur.fetchone()[0])

    # Basic reads for Streamlit
    def list_logs(self, limit: int = 100, offset: int = 0, question_type: Optional[str] = None, guardrail: Optional[bool] = None):
        where = []
        params = []
        if question_type:
            where.append("question_type = %s")
            params.append(question_type)
        if guardrail is not None:
            where.append("guardrail_triggered = %s")
            params.append(guardrail)
        where_sql = f"WHERE {' AND '.join(where)}" if where else ""
        params.extend([limit, offset])
        with self.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, created_at, filepath, user_prompt, station_id, question_type,
                       guardrail_triggered, total_input_tokens, total_output_tokens
                FROM llm_logs
                {where_sql}
                ORDER BY id DESC
                LIMIT %s OFFSET %s;
                """,
                tuple(params),
            )
            rows = cur.fetchall()
        return [
            {
                "id": r[0],
                "created_at": r[1],
                "filepath": r[2],
                "user_prompt": r[3],
                "station_id": r[4],
                "question_type": r[5],
                "guardrail_triggered": r[6],
                "total_input_tokens": r[7],
                "total_output_tokens": r[8],
            }
            for r in rows
        ]

    def get_log(self, log_id: int):
        with self.cursor() as cur:
            cur.execute(
                """
                SELECT id, created_at, filepath, agent_name, provider, model, user_prompt,
                       station_id, state_filter, time_window, question_type,
                       guardrail_triggered, guardrail_reason,
                       total_input_tokens, total_output_tokens,
                       assistant_answer, tools_json, raw_json
                FROM llm_logs
                WHERE id = %s;
                """,
                (log_id,),
            )
            row = cur.fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "created_at": row[1],
            "filepath": row[2],
            "agent_name": row[3],
            "provider": row[4],
            "model": row[5],
            "user_prompt": row[6],
            "station_id": row[7],
            "state_filter": row[8],
            "time_window": row[9],
            "question_type": row[10],
            "guardrail_triggered": row[11],
            "guardrail_reason": row[12],
            "total_input_tokens": row[13],
            "total_output_tokens": row[14],
            "assistant_answer": row[15],
            "tools_json": row[16],
            "raw_json": row[17],
        }

    def get_checks(self, log_id: int):
        with self.cursor() as cur:
            cur.execute(
                """
                SELECT check_name, passed, score, details, created_at
                FROM eval_checks
                WHERE log_id = %s
                ORDER BY id ASC;
                """,
                (log_id,),
            )
            rows = cur.fetchall()
        return [
            {
                "check_name": r[0],
                "passed": r[1],
                "score": r[2],
                "details": r[3],
                "created_at": r[4],
            }
            for r in rows
        ]

    def get_feedback(self, log_id: int):
        with self.cursor() as cur:
            cur.execute(
                """
                SELECT is_good, comments, reference_answer, created_at
                FROM feedback
                WHERE log_id = %s
                ORDER BY id DESC;
                """,
                (log_id,),
            )
            rows = cur.fetchall()
        return [
            {
                "is_good": r[0],
                "comments": r[1],
                "reference_answer": r[2],
                "created_at": r[3],
            }
            for r in rows
        ]

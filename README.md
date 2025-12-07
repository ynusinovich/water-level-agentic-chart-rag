# Water Level Agentic Chart RAG

Agent that answers questions about USGS water-level charts by:
- Finding stations via Qdrant semantic search
- Extracting Highcharts/Plotly series via Playwright tools
- Running an LLM to reason about trends/statistics
- Logging every run with guardrails + monitoring dashboards

## Architecture
- **Ingestion**: `scripts/ingest_data.py` fetches USGS stations for AZ/CA/CO/NM/UT/NV, embeds with OpenAI, and upserts to Qdrant.
- **Agent runtime**: `scripts/usgs_agent.py` (tools + LangChain agent). Always run through `scripts/invoke_agent.py` for guardrails + logging.
- **Guardrails**: `scripts/guardrails.py` (input validation for station/state/time window + output safety).
- **Monitoring**: JSON logs → `monitoring/runner.py` → Postgres → Streamlit UI (`monitoring/app.py`) + Grafana dashboards (`monitoring/grafana/...`).
- **Evals**: `evals/` scripts + CSVs for manual/LLM judging.
- **Tests**: `tests/` (unit + judge-based).

## Prerequisites
- Docker & Docker Compose
- OpenAI API key (required)
- Optional LangSmith key (already wired via env vars)
- Python/uv only if you prefer to run outside Docker; all commands below assume Docker.

## Environment variables (.env)
- `OPENAI_API_KEY` (required)
- `OPENAI_BASE_URL` (optional, default https://api.openai.com/v1)
- `COLLECTION_NAME` (default `usgs_stations`)
- `LANGSMITH_API_KEY` / `LANGSMITH_PROJECT` (optional tracing)
- Monitoring stack uses `DATABASE_URL` via docker-compose (no manual config needed).

## Quickstart
1. Build image and start core services:
   - `make build`
   - `make up`
2. Ingest stations:
   - `make ingest`
3. Start monitoring stack:
   - `make monitoring-up`
4. Ask a few questions in the CLI (see "Example Questions" section, below):
   - `make cli`
5. Open monitoring UIs:
   - Streamlit at http://localhost:8501
   - Grafana at http://localhost:3000
6. (Optional) Run tests:
   - `make test`

## Ingestion
Populate Qdrant with Southwest stations:
```bash
make ingest
```

## Monitoring & Guardrails
- Start monitoring stack:
```bash
make monitoring-up
```
- Logs are written to `logs/` (JSON). `log-poller` ingests into Postgres.
- UIs:
  - Streamlit monitoring: http://localhost:8501
  - Grafana: http://localhost:3000 (admin/admin). Dashboards auto-provisioned from `monitoring/grafana/dashboards`.
- Guardrails are always active via `scripts/invoke_agent.py` (used by `scripts/usgs_agent.py`).

## Running the agent
Interactive CLI:
```bash
make cli
```
(Internally calls `invoke_agent`, which runs guardrails, logs, and tags LangSmith.)

## Example questions
- “Give me a quick snapshot of current water levels at station 09421500 for the last 24 hours.”
- “Find a nearby station to Bagdad, AZ with recent non-zero water levels and summarize the last 24 hours.”
- “Compare the last 48 hours of water level at stations 09180500 and 09179000. Which one is rising faster?”
- “For Colorado, pick a surface-water station with at least a week of recent data and summarize the 7-day trend.”
- “Tell me whether station 09421500 has gone above 5 feet at any point in the last 7 days.”
These are just examples; the agent should work for other reasonable USGS water-level questions within AZ, CA, CO, NM, UT, and NV.

## Testing
Run inside the app container:
```bash
make test  # all tests
make test-unit  # unit tests only
make test-judge  # judge-based tests
```

## Services & Ports
- Qdrant: http://localhost:6333 (API), 6334 (gRPC)
- Monitoring Postgres: exposed on 5432
- Streamlit monitoring app: http://localhost:8501
- Grafana: http://localhost:3000 (admin/admin)

## Project Structure (key paths)
```
scripts/  # ingest_data.py, usgs_agent.py, invoke_agent.py, guardrails.py
monitoring/  # logging, callbacks, db, evaluator, runner, Streamlit app, Grafana provisioning
evals/  # questions, manual eval logs, ground-truth builder
tests/  # unit + judge-based tests
docker-compose.yml  # app, qdrant, monitoring stack
Dockerfile.dev  # dev image with uv + Playwright
```

## Remove data and services

To stop the services and remove this project’s containers and volumes from your system:

Using the Makefile:
```
make monitoring-down  # stop monitoring services
make down  # stop app & Qdrant
make reset-data  # stop everything and remove project volumes
```
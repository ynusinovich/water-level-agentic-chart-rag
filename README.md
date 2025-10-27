# Water Level Agentic Chart RAG [IN PROGRESS]

An agentic RAG system for analyzing USGS water monitoring data via semantic search, with ingestion into Qdrant and charts/analysis powered by Plotly.

## Quick Start

### Prerequisites
- Docker and Docker Compose
- An OpenAI API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/ynusinovich/water-level-agentic-chart-rag.git
cd water-level-agentic-chart-rag
```

2. Create your environment file:
```bash
cp .env.example .env
# edit .env and set:
# OPENAI_API_KEY=sk-...
# (you can also set COLLECTION_NAME / OPENAI_BASE_URL here if you want)
```

3. Build the app image:
```bash
docker compose build app
```

4. Start Qdrant vector database:
```bash
docker compose up -d qdrant
```

### Data Ingestion

Run the initial data ingestion:
```bash
docker compose run --rm app python scripts/ingest_data.py
```

This will:
- Fetch USGS monitoring station metadata
- Create embeddings for semantic search
- Store in Qdrant vector database

### Testing Search

Test the semantic search functionality:
```bash
docker compose run --rm app python scripts/test_search.py
```

## Verifying Functionality

### Qdrant readiness:
```bash
curl -s http://localhost:6333/readyz
```

### List collections:
```bash
curl -s http://localhost:6333/collections | jq
```

### Inspect collection:
```bash
curl -s http://localhost:6333/collections/water_stations | jq
```

## Developer Workflow

### Run any script in the container
```bash
docker compose run --rm app python <your_script>.py
```

### Open a shell in the app container
```bash
docker compose run --rm app bash
```

### Regenerate / update the dependency lock
```bash
docker compose run --rm app uv pip compile pyproject.toml -o requirements.lock
```

## Data Management

### Full reset (wipe volumes)

```bash
docker compose down -v
docker compose up -d qdrant
docker compose run --rm app python ingest_data.py
```

### Reingest without restarting Docker

```bash
docker compose run --rm app python ingest_data.py
```

## Verify Data

Check if data was ingested correctly:
```bash
# Check Qdrant is running
curl http://localhost:6333/collections

# Check collection details
curl http://localhost:6333/collections/usgs_stations
```

## Configuration
- Environment variables (from .env and/or docker-compose.yml):

- OPENAI_API_KEY – required

- OPENAI_BASE_URL – defaults to https://api.openai.com/v1

- COLLECTION_NAME – defaults to water_stations

- QDRANT_HOST – set by Compose to qdrant (container name)

- QDRANT_PORT – defaults to 6333

- Command-line options:
    - --iv-only (optional): restricts USGS stations to sites that have instantaneous values (IV).
Omit this to include more groundwater sites that often lack IV.

## Project Structure
```
water-level-agentic-chart-rag/
├── docker-compose.yml  # qdrant (pinned) + app (uv-based)
├── Dockerfile.dev  # builds app image, creates uv lock, installs from loc
├── .env.example  # template for env vars
├── requirements.lock  # generated inside container
├── README.md  # this file
├── pyproject.toml  # dependencies (source of truth)
├── scripts/
│   ├── ingest_data.py  # fetch + transform + embed + upsert into Qdrant
│   └── test_search.py  # quick semantic search test
└── qdrant_storage/  # Qdrant persistent data (created by Compose)
```

## Troubleshooting

### OpenAI API Key Error
Ensure your `.env` file contains:
```
OPENAI_API_KEY=sk-...your-key-here...
```

### Qdrant Connection Error
Check if Qdrant is running:
```bash
docker compose ps
```

If not running, start it:
```bash
docker compose up -d qdrant
```

### Rate Limiting
The ingestion script includes delays to respect API rate limits. If you encounter rate limit errors:
- For USGS: The script already includes 0.5s delays
- For OpenAI: Reduce batch size in `ingest_data.py` (default is 50)

### Memory Issues
If running on a low-memory system:
- Reduce the batch size in `ingest_data.py`
- Ingest fewer states at once (modify the `states` list)
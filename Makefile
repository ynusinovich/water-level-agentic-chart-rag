.PHONY: help build up down logs ingest cli test test-unit test-judge monitoring-up monitoring-down reset-data

help:
	@echo "Available targets:"
	@echo "  build            - Build the app image"
	@echo "  up               - Start core services (qdrant + app)"
	@echo "  down             - Stop all services"
	@echo "  logs             - Tail app service logs"
	@echo "  ingest           - Run station ingestion"
	@echo "  cli              - Run the app CLI"
	@echo "  test             - Run all tests"
	@echo "  test-unit        - Run unit tests only"
	@echo "  test-judge       - Run judge-based tests only"
	@echo "  monitoring-up    - Start monitoring stack (postgres, log-poller, monitoring-app, grafana)"
	@echo "  monitoring-down  - Stop monitoring stack services"
	@echo "  reset-data       - Remove all app data"

build:
	docker compose build app

up:
	docker compose up -d qdrant app

down:
	docker compose down

logs:
	docker compose logs -f app

ingest:
	docker compose run --rm app python -m scripts.ingest_data

cli:
	docker compose run --rm app python -m scripts.usgs_agent

test:
	docker compose run --rm app pytest

test-unit:
	docker compose run --rm app pytest tests/unit

test-judge:
	docker compose run --rm app pytest tests/judge

monitoring-up:
	docker compose up -d postgres log-poller monitoring-app grafana

monitoring-down:
	docker compose stop postgres log-poller monitoring-app grafana

reset-data:
	docker compose down --volumes --remove-orphans

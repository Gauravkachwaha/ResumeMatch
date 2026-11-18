<div align="center">

# JobPilot

**An AI job-search platform with coordinated agents, resume evidence, and human approval.**

![Version](https://img.shields.io/badge/version-0.4.0-2563EB?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C?style=flat-square)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=flat-square&logo=postgresql&logoColor=white)

[Capabilities](#core-capabilities) · [Architecture](#architecture) · [Quick start](#quick-start) · [API reference](#main-apis) · [Documentation](#documentation)

</div>

---

JobPilot is a portfolio-oriented AI job-search platform built with **LangGraph**, **Browser Use**, and **Resume RAG**. It coordinates planning, job discovery, resume analysis, ranking, and approved application execution, backed by persistent memory and an execution runtime with durable task storage.

**New in v0.4.0:** A durable queue and worker model, PostgreSQL application storage, token and cost accounting, failure classification, an offline agent benchmark, OpenTelemetry integration, and CI.

## Core capabilities

| Capability | What it provides |
| --- | --- |
| **Multi-agent orchestration** | Planner, Resume, Search, Ranking, Browser, and Evaluation agents coordinated through LangGraph. |
| **Human approval** | Application tasks remain at `waiting_approval` until explicitly approved. |
| **Browser automation** | Browser Use powers job discovery, research, and approved application execution. |
| **Resume RAG** | TXT, Markdown, PDF, and DOCX parsing with section-aware chunks and job-specific evidence retrieval. |
| **Persistent memory** | User preferences, job lifecycle state, and LangGraph checkpoints with a stable `workflow_thread_id`. |
| **Durable execution** | Queue-backed tasks with worker leases, delayed retries, and dead-letter state. |
| **Flexible storage** | PostgreSQL and pgvector support, with SQLite application, vector, and checkpoint storage for local use. |
| **Usage and observability** | Token accounting, configurable cost estimates, browser execution metrics, and optional OpenTelemetry traces. |
| **Evaluation and CI** | A deterministic offline benchmark plus compilation, tests, and benchmark execution on Python 3.11 and 3.13. |

## Architecture

```text
                         Web UI / API
                              │
                           FastAPI
                              │
                      Durable task queue
                              │
                      Worker claim + lease
                              │
                       LangGraph Planner
                              │
               ┌──────────┬─────────┬──────────┐
               ▼          ▼         ▼          ▼
             Resume     Search    Ranking    Browser
               └──────────┴─────────┴──────────┘
                              │
                          Evaluation
                              │
                       Pass / Replan
                              │
                        Terminal state
```

The planner selects an agent sequence for each task. Evaluation determines whether the workflow passes or needs replanning.

| Persistence layer | PostgreSQL deployment | Local default |
| --- | --- | --- |
| Application data | PostgreSQL: jobs, tasks, traces, memory, queue, usage, and benchmarks | SQLite |
| Resume vectors | pgvector | SQLite with cosine search |
| Workflow checkpoints | PostgreSQL checkpointer | SQLite checkpointer |

### Durable execution

```text
POST /api/tasks → Persist task → Approval gate, when required → Enqueue
                                                                 │
                                                        Worker claim + lease
                                                                 │
                                                        LangGraph workflow
                                                                 │
                              ┌──────────────────────────────────┼─────────────────────────┐
                              ▼                                  ▼                         ▼
                           Success                         Agent failure             Runtime failure
                              │                                  │                         │
                             Done                         Evaluate / Replan        Delayed retry / Dead
```

Application tasks enter the queue only after approval. Agent-level recovery happens within the workflow; runtime failures are handled by worker retries.

## Quick start

**Requirements:** Python 3.11+ for local execution, or Docker Compose for the container stack. Run commands from the project root.

### 1. Create a virtual environment

```bash
python -m venv .venv
```

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

### 2. Install dependencies and configure the environment

```bash
python -m pip install -r requirements.txt
```

Copy the configuration template:

```powershell
# Windows PowerShell
Copy-Item .env.example .env
```

```bash
# macOS / Linux
cp .env.example .env
```

Edit `.env` for the capabilities you want to use:

| Setting | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Enables provider-backed LLM and embedding requests. |
| `OPENAI_BASE_URL` | Optional endpoint for an OpenAI-compatible provider. |
| `LLM_MODEL` / `EMBEDDING_MODEL` | Selects the language and embedding models. |
| `BROWSER_USE_API_KEY` | Enables the Browser Use model; the browser agent can alternatively use `OPENAI_API_KEY`. |
| `WORKER_ENABLED` | Enables the embedded worker; defaults to `true` for local execution. |

Local fallback paths support offline development, including feature-hash embeddings. Live browser tasks require a configured model provider. See [`.env.example`](.env.example) for all storage, worker, pricing, and telemetry settings.

### 3. Start the application

```bash
python -m uvicorn app.main:app --reload --port 8000
```

| Local endpoint | Purpose |
| --- | --- |
| [localhost:8000](http://localhost:8000) | Web interface |
| [localhost:8000/docs](http://localhost:8000/docs) | Interactive API documentation |
| [localhost:8000/api/health](http://localhost:8000/api/health) | Health and configuration status |

The embedded worker starts with the API when `WORKER_ENABLED=true`.

## Docker deployment

With Docker Compose installed and `.env` configured as above:

```bash
docker compose up --build
```

The included stack starts three services:

| Service | Responsibility |
| --- | --- |
| `postgres` | PostgreSQL with pgvector and a persistent database volume. |
| `jobpilot` | FastAPI web process exposed on port 8000, with the embedded worker disabled. |
| `worker` | Independent queue worker consuming tasks from the shared database. |

See the [production runtime guide](docs/PRODUCTION.md) for queue semantics, storage configuration, and telemetry setup.

## Main APIs

### Planning and tasks

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `POST` | `/api/plan` | Preview an agent plan. |
| `POST` | `/api/tasks` | Create a task. |
| `POST` | `/api/tasks/{task_id}/approve` | Submit an approval decision. |
| `POST` | `/api/tasks/{task_id}/resume` | Resume a task. |
| `GET` | `/api/tasks/{task_id}` | Inspect task details. |
| `GET` | `/api/tasks` | List tasks. |

### Resume retrieval and memory

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `POST` | `/api/resume/index` | Parse and index a resume. |
| `POST` | `/api/resume/search` | Retrieve relevant resume evidence. |
| `GET` | `/api/memory/users/{user_id}` | Retrieve user preferences and job memory. |
| `PUT` | `/api/memory/users/{user_id}/{key}` | Update a user preference. |
| `POST` | `/api/memory/jobs` | Record job lifecycle state. |

### Operations and evaluation

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/queue` | Inspect the task queue. |
| `GET` | `/api/metrics/summary` | View operational metrics. |
| `GET` | `/api/tasks/{task_id}/usage` | Inspect task usage and estimated costs. |
| `POST` | `/api/benchmarks/run` | Run the core benchmark. |
| `GET` | `/api/benchmarks` | Retrieve benchmark history. |

Request schemas and additional endpoints are available in the [interactive API documentation](http://localhost:8000/docs) when the application is running.

## Usage, costs, and observability

Provider-reported LLM and embedding usage is stored in `usage_events`. Browser tasks record duration, steps, actions, and errors. Cost estimates use the prices you configure:

```dotenv
LLM_INPUT_COST_PER_1M=0
LLM_OUTPUT_COST_PER_1M=0
EMBEDDING_COST_PER_1M=0
```

Set these values for your selected provider and models; zero defaults produce zero estimated token costs. Provider pricing is not hard-coded.

Database-backed metrics are available through `/api/metrics/summary`. Optional OpenTelemetry export is configured through `OTEL_ENABLED` and `OTEL_EXPORTER_OTLP_ENDPOINT`.

Failure classification covers timeout, network, rate-limit, CAPTCHA, authentication, element, model, validation, safety-block, and unknown errors.

## Tests and offline benchmark

Install the test dependency and run the suite:

```bash
python -m pip install pytest
python -m pytest -q
```

Run the deterministic benchmark:

```bash
python -m app.evaluation.benchmark
```

The benchmark checks planner routing for job searches and applications, evaluation quality-gate behavior, and timeout/CAPTCHA failure classification. Results are persisted in `benchmark_runs`.

GitHub Actions runs compilation, tests, and the offline benchmark on **Python 3.11 and 3.13**. See the [benchmark guide](docs/BENCHMARK.md) for coverage details.

## Repository map

```text
app/
├── agents/              # Planner, Resume, Search, Ranking, Browser
├── evaluation/          # Offline agent benchmark
├── memory/              # User preferences and job lifecycle memory
├── observability/       # OpenTelemetry integration
├── rag/                 # Document parsing, chunking, and vector retrieval
├── runtime/             # Durable queue worker
├── services/            # LLM, embeddings, usage, and failure classification
├── static/              # Web interface assets
├── storage/             # SQLite/PostgreSQL persistence and task queue
├── workflow/            # LangGraph orchestration and checkpoints
├── config.py            # Application settings
├── db.py                # Storage compatibility facade
└── main.py              # FastAPI entry point

benchmarks/              # Benchmark resources
docs/                    # Architecture and runtime documentation
tests/                   # Automated test suite
.github/workflows/       # Continuous integration
Dockerfile               # Application container
docker-compose.yml       # API, worker, and PostgreSQL stack
.env.example             # Environment configuration template
```

## Documentation

| Guide | Contents |
| --- | --- |
| [Architecture](docs/ARCHITECTURE.md) | Agent coordination, retrieval, memory, and workflow design. |
| [Production runtime](docs/PRODUCTION.md) | Durable execution, PostgreSQL, accounting, and telemetry. |
| [Agent benchmark](docs/BENCHMARK.md) | Offline evaluation cases and persisted results. |

### Validation scope

The automated suite targets the SQLite/local path without external services. PostgreSQL/pgvector, live Browser Use, and OTLP export require their corresponding services and credentials and need integration validation before production reliability claims.

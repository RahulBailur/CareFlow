# CareFlow — Voice-First Patient & Appointment Assistant (POC)

A proof-of-concept outpatient platform: patients book appointments and track a live queue, doctors manage their queue and broadcast delays, and **CareBot**, a multi-agent voice assistant, handles booking, symptom-based department routing, records lookups and hospital FAQs in English, Hindi and Kannada.

The engineering focus is the voice agent: two interchangeable voice pipelines, measured per-stage latency, graceful failover when free-tier quotas run out, and guardrails enforced in code. Everything runs on free tiers or self-hosted open-source components.

> **Scope:** This is a POC, not a production system. It is built to demonstrate agent architecture, latency engineering and reliability patterns on a small scale. Production concerns that don't change the architecture are listed under [Future Work](#-future-work-out-of-poc-scope).

## Problem Statement

Outpatient departments run on front-desk phone calls and paper tokens. Patients don't know which department to visit, can't see real wait times, and call the front desk for information a system could answer. CareFlow addresses this with real-time queue state, self-service booking, and a voice/text assistant that handles routine requests, while keeping clinical judgement strictly with doctors.

---

## 🚀 Features

### 🧑‍⚕️ Patient
- Book, reschedule, and cancel appointments by department or doctor
- Live queue position ("3 patients ahead of you") and doctor delay broadcasts with updated ETA
- Symptom-based department routing (**not diagnosis** — "chest pain" → Cardiology, with a disclaimer on every routing response)
- Visit & prescription history (view-only, last 12 months)
- Talk to CareBot by voice or text, in English, Hindi, Kannada or code-mixed speech ("kal ka appointment cancel karo")

### 👨‍⚕️ Doctor
- Today's queue — mark patient `in_consultation` / `done` / `no_show`
- Broadcast delay status (pushed live to every waiting patient via Socket.IO)
- Schedules are seeded; editing them is out of POC scope

### 🏥 Admin — Latency & Analytics page
- Department load (queue depth, doctors on duty, today's appointments)
- Per-stage voice latency (P50 / P95) for each pipeline, time to first audio
- Intent mix, fallback rate (how often each failover tier was used), Gemini quota errors

### 🤖 CareBot — multi-agent voice assistant

**Specialist agents** (each bound to a narrow tool set):
- **Booking Agent** — check availability, book / reschedule / cancel (real DB tool calls)
- **Triage Agent** — symptoms → department suggestion only; never diagnoses, prescribes or gives dosage
- **Records Agent** — the authenticated patient's own history only
- **Support Agent** — timings, department locations, emergency contact

**Routing:** a fast local intent classifier (keywords + a small multilingual embedding model) handles most turns with no model call. Only low-confidence turns go to the LLM for classification. This removes one LLM round-trip from the typical turn.

**Two voice pipelines** (switchable in the UI for side-by-side demos):

| | Pipeline A — Real-time (primary) | Pipeline B — Cascaded, open source (fallback + baseline) |
|---|---|---|
| Flow | Browser mic → FastAPI WebSocket → **Gemini Live API** (native audio in/out, function calling) → audio streamed back | **Silero VAD** → **faster-whisper** STT → intent classifier → **Gemini Flash-Lite** (streamed) → sentence-chunked **Gemini TTS** / **Piper** → streamed playback |
| Strength | Lowest latency; speech-to-speech in one model | Every stage measurable and swappable; works when Gemini Live quota is exhausted |
| Routing | One Live session holds all tools; routing is a tool-scoped system prompt. Guardrail checks run server-side on every tool result | Per-turn router dispatches to specialist agents |

The difference in how routing works between A and B is a deliberate trade-off, documented in the README: the Live API keeps a single session per conversation, so per-turn agent hand-off isn't possible there without restarting the session and losing latency.

**Shared CareBot features:**
- **Conversation memory** — MongoDB, keyed per session, so follow-ups ("what time was that again?") resolve
- **Conversation state machine** (IDLE / LISTENING / TRANSCRIBING / THINKING / TOOL_EXECUTION / GENERATING_RESPONSE / SPEAKING / WAITING_FOR_NEXT_INPUT / ERROR / END) streamed over Socket.IO to drive the UI — no guessed client-side timing
- **Barge-in** — user speech during playback stops TTS and starts a new turn
- **Redis workflow tracker** — each booking moves through `received → intent_classified → slot_checked → confirmed/failed`
- **Redis response cache** for frequent questions ("OPD timings") — no LLM call
- **Per-turn analytics** — pipeline used, intent, per-stage timings, fallback tier, logged to MongoDB

### 🔐 Auth
- JWT (24-hour expiry), bcrypt password hashing
- **Patients self-register; doctor and admin accounts are seeded only** (no public privileged registration)
- Indian phone validation (10-digit); doctor medical registration number on seeded profiles

---

## 🛠 Tech Stack

Every component is free-tier or open source.

| Layer | Technology | Notes |
|---|---|---|
| Frontend | React 18, React Router 6, Socket.io-client, Axios, Recharts | Built and served as static files by FastAPI (one container) |
| Backend | Python 3.12, FastAPI, Uvicorn, python-socketio | Native FastAPI WebSockets for audio, Socket.IO for queue/state events |
| Database | MongoDB (Motor + Beanie) | Docker locally · **MongoDB Atlas free tier** for the demo · mongomock-motor for tests |
| Cache / Workflow | Redis | Docker locally · **Upstash free tier** for the demo |
| Auth | PyJWT + bcrypt | |
| LLM (primary) | **Gemini Flash / Flash-Lite** (free tier) | Model IDs configured in `.env` |
| LLM (offline fallback) | **Ollama** with a small local instruct model | Optional Docker Compose profile |
| Real-time voice | **Gemini Live API** native-audio model (free tier) | Pipeline A |
| Intent classifier | Keyword rules + multilingual **sentence-transformers** model | Runs locally on CPU, no API call |
| VAD | **Silero VAD** | Server-side end-of-speech detection, barge-in |
| STT | **faster-whisper** (`small`, int8, CPU) | Pipeline B; handles English and Hindi, Kannada quality to be measured |
| TTS | **Gemini Flash TTS** (free tier) · **Piper** fallback (local, English) | Sentence-by-sentence streaming |
| Rate limiting | slowapi | On chat, voice and auth endpoints |
| Observability | **structlog** (JSON logs, request ID per turn) · optional **Langfuse** (self-hosted) for LLM traces | |
| Testing | pytest, pytest-asyncio, httpx · Jest + React Testing Library | LLM/STT/TTS mocked in tests |
| Evals & load | Custom eval runner · **Locust** | |
| Quality | ruff (lint + format), mypy, ESLint, Prettier, pre-commit, gitleaks | |
| DevOps | Docker, Docker Compose, GitHub Actions, Dependabot | |
| Demo hosting | **Hugging Face Spaces (Docker)** or **Render free tier** | Free instances sleep when idle — warm up before a demo |

> Free-tier limits and model names change. Check each provider's current pricing page before relying on a specific model or quota. Model IDs live only in `.env`, never in code.

---

## 🎙 Voice Architecture & Latency

### Latency budget (Pipeline B, per turn)

| Stage | Measured as | Main levers |
|---|---|---|
| End-of-speech detection | last speech frame → VAD end event | VAD silence threshold (shorter = faster, more cut-offs) |
| STT | VAD end → final transcript | Model size, int8 quantisation, beam size |
| Routing | transcript → agent selected | Local classifier first; LLM only on low confidence |
| LLM time-to-first-token | request → first token | Flash-Lite, short prompts, Redis cache hits skip this stage |
| TTS first chunk | first full sentence → first audio bytes | Sentence-level chunking, stream while LLM still generating |
| **Time to first audio (TTFA)** | **user stops speaking → first audio plays** | **The headline number** |

Pipeline A is measured end to end (user stops speaking → first audio byte), since stages happen inside one model.

### Targets (to be measured, not claimed)

| Metric | Pipeline A | Pipeline B |
|---|---|---|
| TTFA P50 | < 1.5 s | < 2.5 s |
| TTFA P95 | < 2.5 s | < 4.0 s |

Measured on a laptop CPU over a home connection. The README publishes the **actual** P50/P95 per stage, before and after each optimisation (e.g. "TTS sentence streaming cut TTFA from X to Y"). That before/after table is the main deliverable of the project.

### Failover chain

```
Pipeline A (Gemini Live)
  └─ 429 / timeout / session drop ──▶ Pipeline B with Gemini Flash-Lite
                                        └─ 429 / timeout ──▶ Ollama (local)
                                                              └─ unavailable ──▶ rule-based knowledge base
Redis cache is checked before any LLM call.
```

- **Circuit breaker:** after 3 consecutive Gemini 429s, skip Gemini for 60 seconds instead of failing every request.
- Every turn records which tier answered; the fallback rate is shown on the analytics page.
- The Live API limits session duration; the client reconnects transparently and conversation memory carries context across sessions.

### Telephony-quality testing

Gnani-style deployments run over phone lines (8 kHz narrowband). The eval script downsamples test audio to 8 kHz and compares STT word error rate against 16 kHz audio, so the quality drop on phone-grade audio is measured rather than assumed. No telephony provider is needed.

---

## 📁 Project Structure

```
careflow/
├── .github/
│   ├── workflows/ci.yml              # Lint, type-check, tests, guardrails, evals
│   ├── pull_request_template.md
│   └── dependabot.yml
├── backend/
│   ├── models/
│   │   ├── user.py                   # Patient / doctor / admin (role field)
│   │   ├── appointment.py            # Unique index on (doctor_id, slot_start)
│   │   ├── schedule.py               # Seeded doctor slots
│   │   ├── hospital_config.py        # Seeded departments / timings / contacts
│   │   ├── turn_analytics.py         # Per-turn pipeline, intent, stage timings, fallback tier
│   │   └── conversation_turn.py      # Conversation memory
│   ├── routes/
│   │   ├── auth.py                   # Patient register, login
│   │   ├── appointments.py           # Availability, book, reschedule, cancel, history, queue
│   │   ├── doctors.py                # Queue status updates, delay broadcast
│   │   ├── admin.py                  # Department load + analytics aggregation
│   │   ├── chat.py                   # Text turn (HTTP)
│   │   └── voice_ws.py               # WebSocket: /ws/voice?pipeline=live|cascade
│   ├── agents/
│   │   ├── intent_classifier.py      # Keywords + embeddings; LLM only on low confidence
│   │   ├── orchestrator.py           # Dispatch to specialist agents (Pipeline B + text)
│   │   ├── base.py                   # Shared tool-calling loop
│   │   ├── booking_agent.py
│   │   ├── triage_agent.py
│   │   ├── records_agent.py
│   │   ├── support_agent.py
│   │   ├── general_agent.py          # Rule-based knowledge base (last fallback tier)
│   │   ├── live_session.py           # Gemini Live session: tool declarations + server-side execution
│   │   └── guardrails.py             # Diagnosis/dosage output filter, disclaimer injection
│   ├── tools/
│   │   ├── booking_tools.py          # Identity injected from session, never from model args
│   │   ├── triage_tools.py
│   │   ├── records_tools.py
│   │   └── support_tools.py
│   ├── voice/
│   │   ├── live_pipeline.py          # Pipeline A — browser ↔ server ↔ Gemini Live
│   │   ├── cascade_pipeline.py       # Pipeline B — VAD → STT → agent → TTS, streamed
│   │   ├── vad.py                    # Silero VAD wrapper, barge-in detection
│   │   ├── audio_utils.py            # Resampling (16 kHz / 8 kHz), PCM framing
│   │   └── timing.py                 # Per-stage timers → turn_analytics
│   ├── services/
│   │   ├── llm.py                    # Provider abstraction: Gemini / Ollama / Mock
│   │   ├── stt.py                    # faster-whisper / Mock
│   │   ├── tts.py                    # Gemini TTS / Piper / Mock
│   │   ├── failover.py               # Fallback chain + circuit breaker
│   │   └── redis_client.py           # Cache + workflow tracker
│   ├── memory/conversation_memory.py
│   ├── scripts/
│   │   ├── seed_data.py              # Synthetic patients, doctors, schedules, appointments
│   │   └── make_eval_audio.py        # Synthesises eval audio via TTS (no real voices)
│   ├── eval/
│   │   ├── utterances.jsonl          # 60–80 labelled utterances: EN / HI / KN / code-mixed
│   │   ├── run_eval.py               # Intent accuracy, tool-call correctness
│   │   ├── stt_wer.py                # WER at 16 kHz vs 8 kHz
│   │   └── latency_bench.py          # Runs N turns per pipeline, reports P50/P95 per stage
│   ├── loadtest/locustfile.py        # 20–50 concurrent text sessions
│   ├── tests/
│   │   ├── conftest.py
│   │   ├── test_auth.py              # Incl. "cannot self-register as doctor/admin"
│   │   ├── test_appointments.py      # Incl. concurrent double-booking test
│   │   ├── test_doctors.py
│   │   ├── test_failover.py          # Circuit breaker + fallback order
│   │   ├── test_sockets.py           # Unauthenticated socket rejected; patient sees own position only
│   │   ├── test_intent_classifier.py
│   │   ├── test_records_scoping.py   # Guardrail
│   │   ├── test_identity_injection.py# Guardrail: tools ignore model-supplied IDs
│   │   └── test_triage_guardrails.py # Guardrail
│   ├── auth_utils.py
│   ├── config.py
│   ├── database.py
│   ├── sockets.py                    # JWT-authenticated Socket.IO: queue + voice state events
│   ├── main.py                       # Also serves the built frontend
│   ├── pyproject.toml
│   └── Dockerfile                    # Multi-stage: build frontend, then backend image
├── frontend/
│   └── src/
│       ├── context/AuthContext.js
│       ├── components/
│       │   ├── Navbar.js
│       │   ├── DemoBanner.js             # "Demo — do not enter real health information"
│       │   ├── QueueTracker.js
│       │   ├── ChatBot.js                # Text + voice, pipeline A/B toggle
│       │   ├── VoiceStream.js            # Mic capture, PCM streaming, playback, barge-in
│       │   ├── TriageDisclaimer.js
│       │   └── voiceConversationMachine.js
│       ├── pages/
│       │   ├── Login.js / Register.js
│       │   ├── PatientDashboard.js
│       │   ├── BookAppointment.js
│       │   ├── VisitHistory.js
│       │   ├── DoctorDashboard.js        # Queue controls + delay broadcast
│       │   └── AnalyticsDashboard.js     # Admin: department load + latency + fallbacks
│       └── __tests__/
├── docker-compose.yml                # app + mongo + redis (+ ollama via --profile offline)
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── README.md
└── SECURITY.md
```

---

## 🔐 Roles & Access

| Role | How created | Capabilities |
|---|---|---|
| Patient | Self-registration | Book/reschedule/cancel, live queue, CareBot, own history |
| Doctor | Seeded only | Queue controls, delay broadcast |
| Admin | Seeded only | Department load, latency & analytics page |

## 🌐 API Endpoints

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| POST | `/api/auth/register` | — | Register a **patient** (role is not a parameter) |
| POST | `/api/auth/login` | — | Login (all roles) |
| GET | `/api/appointments/availability` | JWT | Open slots by department/doctor/date |
| POST | `/api/appointments/book` | JWT (patient) | Book a slot (409 if already taken) |
| PUT | `/api/appointments/{id}` | JWT (owner) | Reschedule / cancel |
| GET | `/api/appointments/me` | JWT (patient) | Own visit & prescription history (12 months) |
| GET | `/api/appointments/queue/{doctor_id}` | JWT | Queue status (patients see only their own position) |
| PUT | `/api/appointments/{id}/status` | JWT (doctor) | `in_consultation` / `done` / `no_show` |
| GET | `/api/doctors/{doctor_id}/schedule` | JWT | View schedule |
| POST | `/api/doctors/status` | JWT (doctor) | Broadcast delay / availability |
| GET | `/api/admin/stats` | JWT (admin) | Department load |
| GET | `/api/admin/analytics` | JWT (admin) | Latency P50/P95 per stage & pipeline, intent mix, fallback rate |
| GET | `/api/hospital/config` | — | Public hospital info |
| POST | `/api/chat` | JWT | CareBot text turn (rate-limited) |
| WS | `/ws/voice?pipeline=live\|cascade` | JWT (query/first message) | CareBot voice stream (rate-limited per user) |

---

## ⚠️ Healthcare Guardrails & Security

Guardrails are enforced in code, not only in prompts, and each one has a test.

| Rule | Enforcement | Verified by |
|---|---|---|
| Triage never diagnoses, prescribes or gives dosage | System prompt + pattern-based output filter in `guardrails.py` that replaces violations with a safe fallback; also applied to Live API tool results and transcripts | `test_triage_guardrails.py` |
| Every triage response shows a disclaimer | Added server-side and rendered by `TriageDisclaimer.js` | `test_triage_guardrails.py` + frontend test |
| Identity comes from the session, never the model | Every tool receives `user_id` / role from the verified JWT; tools take no patient/user ID arguments from the LLM. Covers Booking and Records in both pipelines | `test_records_scoping.py`, `test_identity_injection.py` |
| No privileged self-registration | Register endpoint creates patients only; doctors/admins seeded | `test_auth.py` |
| Socket.IO and voice WebSocket require auth | JWT checked on connect; patients receive only their own queue position | `test_sockets.py` |
| No double booking | Unique index on `(doctor_id, slot_start)`; duplicate insert returns 409 | `test_appointments.py` (concurrent requests) |
| Abuse / quota protection | slowapi rate limits on auth, chat and voice | `test_auth.py`, `test_failover.py` |
| Mock DB never used outside tests | App refuses to start with `USE_MOCK_DB=true` unless `ENVIRONMENT=test` | startup check |

Guardrail tests are marked `@pytest.mark.guardrail` and run as their own required CI job.

### Data handling

- **Gemini's free tier allows Google to use submitted content to improve its products.** CareFlow therefore uses **synthetic data only**: seeded patients, TTS-generated eval audio, no real recordings.
- A persistent `DemoBanner` tells users not to enter real health information; the README states the same.
- Raw audio is processed in memory and never stored. Analytics store timings and intent labels, not transcripts.

---

## 💻 Local Development

```bash
git clone https://github.com/RahulBailur/CareFlow.git
cd CareFlow
cp .env.example .env                       # add a free Gemini API key, or use LLM_PROVIDER=mock
docker compose up --build                  # app :8000, mongo :27017, redis :6379
docker compose exec app python scripts/seed_data.py

# Optional: fully offline LLM fallback
docker compose --profile offline up -d ollama
docker compose exec ollama ollama pull <small-instruct-model>
```

`.env.example`:

```dotenv
# Core
ENVIRONMENT=development                    # development | test | demo
JWT_SECRET=change-me-to-a-long-random-string
JWT_EXPIRY_HOURS=24
MONGO_URI=mongodb://mongo:27017/careflow
REDIS_URL=redis://redis:6379/0
USE_MOCK_DB=false

# LLM
LLM_PROVIDER=gemini                        # gemini | ollama | mock
GEMINI_API_KEY=
GEMINI_TEXT_MODEL=                         # current free Flash-Lite model ID
GEMINI_LIVE_MODEL=                         # current free Live native-audio model ID
GEMINI_TTS_MODEL=                          # current free Flash TTS model ID
OLLAMA_URL=http://ollama:11434
OLLAMA_MODEL=

# Voice
VOICE_PIPELINE_DEFAULT=live                # live | cascade
WHISPER_MODEL=small
WHISPER_COMPUTE_TYPE=int8
PIPER_VOICE=en_US-lessac-medium
VAD_SILENCE_MS=500

# Failover
GEMINI_BREAKER_THRESHOLD=3
GEMINI_BREAKER_COOLDOWN_S=60

# Rate limits
RATE_LIMIT_CHAT=20/minute
RATE_LIMIT_VOICE_SESSIONS=5/minute
```

### Demo deployment (free)

| Piece | Service |
|---|---|
| App container (FastAPI + built React) | Hugging Face Spaces (Docker) or Render free |
| MongoDB | MongoDB Atlas free tier |
| Redis | Upstash free tier (`rediss://` URL) |

The POC runs a single Uvicorn worker, so Socket.IO needs no message broker. Scaling beyond one worker is listed under Future Work.

---

## 🧪 Evals & Benchmarks

| Script | What it measures | Runs in |
|---|---|---|
| `eval/run_eval.py --classifier` | Intent accuracy of the local classifier on all labelled utterances | CI (no API calls) |
| `eval/run_eval.py --live` | Tool-call correctness end-to-end with the real LLM | Local only (uses free quota) |
| `eval/stt_wer.py` | faster-whisper WER at 16 kHz vs 8 kHz, per language | Local |
| `eval/latency_bench.py` | P50/P95 per stage for Pipelines A and B over N turns | Local |
| `loadtest/locustfile.py` | P95 response time for 20–50 concurrent text sessions (LLM mocked and real) | Local |

Results go into a **Benchmarks** section of the README: tables, the hardware used, and before/after numbers for each optimisation.

---

## 🌿 Git & GitHub Workflow (Solo, POC-level)

Light but disciplined: protected `main`, every change through a PR with green CI, readable history.

### 1. Repository setup

```bash
mkdir careflow && cd careflow
git init -b main
git config user.name  "Rahul"
git config user.email "<your-github-email>"
git add .
git commit -m "chore: initial project scaffold"
git remote add origin https://github.com/RahulBailur/CareFlow.git
git push -u origin main
```

Make the repo public once the README, guardrails and CI are in place.

**`.gitignore` essentials:**

```gitignore
# Secrets
.env
.env.*
!.env.example

# Python
__pycache__/
*.py[cod]
.venv/
.pytest_cache/
.mypy_cache/
.ruff_cache/

# Node
node_modules/
frontend/build/

# Models, audio and data — never committed
models/
*.onnx
*.bin
*.webm
*.wav
*.mp3
eval/audio/                 # regenerated by scripts/make_eval_audio.py
dumps/
*.log

# OS / editor
.DS_Store
.vscode/
.idea/
```

**Repository files:** `README.md` (overview, architecture diagram, demo GIF, benchmarks, guardrails, setup, CI badge), `LICENSE` (MIT), `SECURITY.md` (demo system, synthetic data only, how to report an issue).

### 2. Branching — GitHub Flow

- `main` is protected; work happens on short-lived branches: `<type>/<issue>-<description>`
- e.g. `feat/14-live-pipeline`, `perf/22-tts-sentence-streaming`, `fix/27-double-booking`

### 3. Commits — Conventional Commits

`<type>(<scope>): <summary>` — types `feat`, `fix`, `perf`, `test`, `refactor`, `docs`, `chore`, `ci`.
Scopes: `auth`, `appointments`, `queue`, `agents`, `voice`, `eval`, `frontend`, `ci`.

```
feat(voice): add Gemini Live pipeline with server-side tool execution
perf(voice): stream TTS per sentence to cut time to first audio
fix(appointments): return 409 on duplicate slot via unique index
```

Use `perf` commits for latency work and put the before/after numbers in the commit body — they feed the README benchmarks.

### 4. Pre-commit hooks

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: end-of-file-fixer
      - id: trailing-whitespace
      - id: check-yaml
      - id: check-added-large-files
        args: ["--maxkb=500"]       # blocks accidentally committed model/audio files
      - id: detect-private-key
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks:
      - id: ruff
        args: ["--fix"]
      - id: ruff-format
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.21.0
    hooks:
      - id: gitleaks
  - repo: https://github.com/compilerla/conventional-pre-commit
    rev: v3.4.0
    hooks:
      - id: conventional-pre-commit
        stages: [commit-msg]
```

```bash
pip install pre-commit
pre-commit install --hook-type pre-commit --hook-type commit-msg
pre-commit autoupdate
```

### 5. Pull requests

- Open a PR for every change; it runs CI and forces a self-review of the diff
- Link the issue (`Closes #14`), **squash and merge**, auto-delete the branch

`.github/pull_request_template.md`:

```markdown
## What / Why
Closes #

## How to test

## Latency impact (voice changes only)
<!-- before → after TTFA / stage timings -->

## Checklist
- [ ] Tests pass locally, guardrail tests pass (`pytest -m guardrail`)
- [ ] No secrets, real patient data, audio or model files committed
- [ ] README / API table updated if behaviour changed
```

### 6. Branch protection (`main`)

Settings → Rules → Rulesets: require a PR (0 approvals — you can't approve your own), require status checks `backend`, `guardrails`, `frontend`, block force pushes and deletion. Rulesets on private repos need GitHub Pro (free with the GitHub Student Developer Pack); otherwise enable them after making the repo public.

### 7. CI — GitHub Actions

`.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true

env:
  ENVIRONMENT: test
  USE_MOCK_DB: "true"
  JWT_SECRET: ci-test-secret
  LLM_PROVIDER: mock              # no real LLM / STT / TTS calls in CI

jobs:
  backend:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: backend
    services:
      redis:
        image: redis:7
        ports: ["6379:6379"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip
      - run: pip install -e ".[dev]"
      - run: ruff check . && ruff format --check .
      - run: mypy .
      - run: pytest -m "not guardrail"
        env:
          REDIS_URL: redis://localhost:6379/0

  guardrails:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: backend
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip
      - run: pip install -e ".[dev]"
      - run: pytest -m guardrail -v
      - run: python eval/run_eval.py --classifier --min-accuracy 0.85

  frontend:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: frontend
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: npm
          cache-dependency-path: frontend/package-lock.json
      - run: npm ci
      - run: npm run lint
      - run: npm test -- --watchAll=false
        env:
          CI: true
      - run: npm run build
```

Set the classifier accuracy gate to your first measured baseline, then raise it as the classifier improves.

### 8. Dependencies and secrets

`.github/dependabot.yml` — monthly, grouped, so it's one PR per ecosystem:

```yaml
version: 2
updates:
  - package-ecosystem: pip
    directory: /backend
    schedule: { interval: monthly }
    groups: { python-deps: { patterns: ["*"] } }
  - package-ecosystem: npm
    directory: /frontend
    schedule: { interval: monthly }
    groups: { npm-deps: { patterns: ["*"] } }
  - package-ecosystem: github-actions
    directory: /
    schedule: { interval: monthly }
```

- Enable **secret scanning + push protection** in Settings → Code security
- The Gemini key lives only in local `.env` and the demo host's secret settings; CI needs no keys
- If a key is ever committed: revoke it in Google AI Studio immediately, then remove it from history

### 9. Issues, milestones, tags

- One GitHub Issue per task, labelled (`feat`, `bug`, `perf`, `guardrail`, `voice`, `agents`, `frontend`)
- One milestone per roadmap phase
- Tag at the end of each milestone — release notes generated from merged PRs:

```bash
git checkout main && git pull
git tag -a v0.4.0 -m "M4: Gemini Live pipeline, failover, latency benchmark"
git push origin v0.4.0
gh release create v0.4.0 --generate-notes
```

### 10. Day-to-day loop

```bash
gh issue create --title "perf: stream TTS per sentence" --label perf,voice --milestone M4
git checkout main && git pull
git checkout -b perf/22-tts-sentence-streaming
# ...code, tests, run latency_bench before and after...
git add -p
git commit -m "perf(voice): stream TTS per sentence"
git push -u origin perf/22-tts-sentence-streaming
gh pr create --fill
gh pr checks --watch
gh pr merge --squash --delete-branch
git checkout main && git pull
```

---

## 🗺 Roadmap (~3 weeks part-time)

| Milestone | Scope | Tag |
|---|---|---|
| M1 — Foundation | Repo, Docker Compose, CI, pre-commit, models, auth (patient register + seeded staff), seed data, demo banner | `v0.1.0` |
| M2 — Appointments & real-time | Availability, booking with unique-slot guarantee, history, authenticated Socket.IO queue, delay broadcast, doctor queue controls | `v0.2.0` |
| M3 — Agents & Pipeline B | Intent classifier, specialist agents, guardrails + tests, eval set, VAD → faster-whisper → Flash-Lite → TTS streaming, barge-in | `v0.3.0` |
| M4 — Pipeline A & reliability | Gemini Live pipeline with server-side tools, failover chain + circuit breaker, latency bench, 8 kHz WER test, Locust run | `v0.4.0` |
| M5 — Analytics & demo | Analytics page, demo deployment, README with architecture diagram, benchmarks, demo GIF and "hardest problem" write-up | `v0.5.0` |

Keep a running engineering log during M3–M4 (what broke, what you measured, what you changed). It becomes the README write-up and the answer to "the most stubborn problem you solved".

---

## 🔭 Future Work (out of POC scope)

- Telephony channel (Twilio / Exotel) — outbound reminder calls with voice confirm/reschedule
- Real-time Indic TTS on GPU (e.g. AI4Bharat Indic Parler-TTS) and Indic-specific STT models
- Multiple Uvicorn workers with the Socket.IO Redis manager
- Refresh tokens, audit log of records access, appointment reminders scheduler
- Doctor schedule/leave editing, complaints, hospital config editing, password reset
- Compliance work (DPDP Act, HIPAA-style controls) and a paid LLM tier so content isn't used for training

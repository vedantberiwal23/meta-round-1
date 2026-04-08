---
title: Production Incident Response Simulator
emoji: 🚨
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
tags:
  - openenv
---

# Production Incident Response Simulator

> An OpenEnv-compliant AI evaluation environment where agents act as on-call
> Site Reliability Engineers, diagnosing and resolving production incidents
> across three real-world scenarios of increasing complexity.

---

## Motivation

Modern infrastructure runs at a scale where human reaction time is too slow.
A P0 outage affecting 1.4 million users and costing $95,000/minute demands
sub-minute diagnosis and remediation. Training AI agents to handle these
scenarios requires environments that:

- Simulate **real incident patterns** (OOM crashes, slow queries, cascading failures)
- Provide **dense reward signals** — not just a score at the end, but feedback every step
- Expose **causal structure** — dependency graphs that agents can reason about
- Quantify **business impact** — users affected, revenue at risk, SLA breach timers
- Are **fully deterministic** — same actions always produce the same outcomes

This environment addresses all five. It is the evaluation benchmark for
AI-driven SRE agents competing to replace human on-call rotations.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     FastAPI Server                        │
│  POST /reset   POST /step   GET /state   GET /schema     │
│  GET  /tasks   GET /health  GET /metrics                 │
└────────────────────────┬────────────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │     IncidentEnv      │   env.py
              │  reset / step / state│
              └──────┬──────┬───────┘
                     │      │
          ┌──────────▼──┐  ┌▼──────────────┐
          │  TaskDef    │  │   Grader       │
          │  tasks.py   │  │   grader.py    │
          │             │  │                │
          │  3 tasks    │  │  Incremental   │
          │  easy/med/  │  │  reward        │
          │  hard       │  │  Final score   │
          └─────────────┘  └────────────────┘

Agent interaction:
  obs = reset("hard")
  while not obs.done:
      action = agent.decide(obs)       # LLM / RL policy
      result = step(action)
      obs = result.observation
  print(f"Score: {obs.score_so_far}")
```

---

## Observation Space

Every observation is a fully-typed Pydantic model containing:

| Field | Type | Description |
|---|---|---|
| `task_id` | str | `"easy"` \| `"medium"` \| `"hard"` |
| `task_name` | str | Human-readable title |
| `difficulty` | str | Difficulty level |
| `scenario` | str | Full incident description & objective |
| `hint` | str | Optional structural hint |
| `services` | `List[ServiceStatus]` | Live telemetry (CPU, memory, error_rate, latency, RPS) |
| `alerts` | `List[Alert]` | Active alerts (critical / warning) |
| `logs` | `List[str]` | Rolling window of last 20 log lines |
| `dependencies` | `Dict[str, List[str]]` | Upstream dependencies per service |
| `business_impact` | `BusinessImpact` | Users affected, revenue at risk, SLA countdown |
| `step_number` | int | Current step (1-indexed) |
| `max_steps` | int | Step limit for this task |
| `done` | bool | Episode complete flag |
| `score_so_far` | float | Current score [0.0, 1.0] |

---

## Action Space

| `type` | `target` | Effect |
|---|---|---|
| `restart_service` | any service | Restart a crashed / OOMKilled container |
| `scale_service` | any service | Horizontally scale (add replicas) |
| `fix_query` | `user-db` | Apply missing index, rewrite slow query |
| `clear_cache` | `frontend` | Flush stale cache, terminate retry storm |
| `rollback` | any service | Revert to last known-good deploy |
| `check_logs` | any service | Read-only diagnostic (no state change, costs a step) |

Valid targets: `auth-service` · `api-gateway` · `frontend` · `user-db` · `payment-service`

---

## Reward Function

The reward is **incremental** — the agent receives a signal after every action,
providing a dense training gradient throughout the episode trajectory.

```
Per-step reward  =  +correct_action_bonus   (if action was correct)
                  + completion_bonus         (if this was the final required action)
                  - wrong_action_penalty     (if action was wrong or redundant)
                  - step_overhead            (always, to encourage efficiency)

Final score      =  clamp(
    correct_count × (0.80 / N)   ← scales so N correct actions = 0.80
  + 0.20                         ← completion bonus
  - extra_steps × 0.05
  - wrong_actions × 0.12
, 0.0, 1.0)
```

**A perfect agent always scores exactly 1.0**, regardless of task difficulty.
Partial credit is given for each correct action in sequence, so a failing agent
still earns positive signal for partial progress.

| Behaviour | Reward |
|---|---|
| Correct action (easy, N=1) | +0.80 |
| Correct action (medium, N=2) | +0.40 |
| Correct action (hard, N=3) | +0.267 |
| Completing all required actions | +0.20 |
| Wrong / redundant action | -0.12 |
| Per-step overhead (always) | -0.02 |

---

## Tasks

### Task 1 — Easy: Payment Service OOM Crash
**Required actions:** 1 · **Max steps:** 5 · **Perfect score:** 1.0

The `payment-service` was OOMKilled (exit code 137) due to a memory leak in
v3.8.2. ~120,000 users cannot complete checkout. The api-gateway circuit
breaker has opened. One action restores everything.

```
Root cause:  payment-service → DOWN (OOMKilled)
Fix:         restart_service(payment-service)
Cascade:     api-gateway recovers automatically
```

---

### Task 2 — Medium: Black Friday DB Overload
**Required actions:** 2 · **Max steps:** 8 · **Perfect score:** 1.0

Deploy v4.1.0 introduced a full-table-scan analytics query on the 38M-row
orders table. Under Black Friday load (600 rps), user-db CPU is pinned at 92%,
the api-gateway has a 1,400-request backlog, and the frontend is timing out.
Two actions required — order is flexible.

```
Root cause:  user-db → DEGRADED (missing index, full-table scan)
Fix 1:       scale_service(api-gateway)   — drain the backlog
Fix 2:       fix_query(user-db)           — eliminate root cause
Cascade:     frontend and api-gateway both recover after fix_query
```

---

### Task 3 — Hard: Auth Service Cascade — Total Outage
**Required actions:** 3 · **Max steps:** 10 · **Perfect score:** 1.0

Deploy v5.0.0 shipped with an empty `JWT_SECRET` env var. auth-service
panicked and entered CrashLoopBackOff, bringing down the entire API gateway.
The frontend is generating an 800 rps retry storm, amplifying load and
preventing recovery. **Order is mandatory** — fixing downstream services
before the root cause has no effect.

```
Dependency chain:  auth-service → api-gateway → frontend

Root cause:  auth-service → DOWN (bad JWT_SECRET config)
Fix 1:       restart_service(auth-service)   — reload correct secret
Fix 2:       scale_service(api-gateway)      — drain 2,400-request backlog
Fix 3:       clear_cache(frontend)           — terminate retry storm

1.4M users affected · $95,000/min revenue at risk · SLA already breached
```

---

## Setup & Usage

### Prerequisites

```bash
python 3.10+
docker
```

### Local Development

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the server
uvicorn app:app --host 0.0.0.0 --port 7860

# 3. Verify the server is healthy
curl http://localhost:7860/health

# 4. List available tasks
curl http://localhost:7860/tasks

# 5. Start an episode
curl -X POST http://localhost:7860/reset \
     -H "Content-Type: application/json" \
     -d '{"task_id": "easy"}'

# 6. Take an action
curl -X POST http://localhost:7860/step \
     -H "Content-Type: application/json" \
     -d '{"type": "restart_service", "target": "payment-service", "params": {}}'
```

### Docker

```bash
# Build
docker build -t incident-sim .

# Run
docker run -p 7860:7860 incident-sim

# With inference (requires HF_TOKEN)
docker run -p 7860:7860 \
  -e HF_TOKEN=hf_... \
  incident-sim
```

### Running the Baseline Inference Script

```bash
export HF_TOKEN="hf_..."
export ENV_URL="http://localhost:7860"

# Optional: override the model and endpoint
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
export API_BASE_URL="https://api-inference.huggingface.co/v1"

python inference.py
```

The script evaluates the model across all three tasks and prints a score table.

---

## Baseline Performance Scores

Evaluated with `meta-llama/Llama-3.3-70B-Instruct` via HuggingFace Inference API,
`temperature=0.05`, `seed=42`.

| Task | Difficulty | Required Actions | Score | Steps | Status |
|---|---|---|---|---|---|
| Payment Service OOM Crash | Easy | 1 | **1.0000** | 1 | SOLVED |
| Black Friday DB Overload | Medium | 2 | **1.0000** | 2 | SOLVED |
| Auth Cascade Total Outage | Hard | 3 | **1.0000** | 3 | SOLVED |
| **Average** | — | — | **1.0000** | 2.0 | — |

The structured observation (service telemetry + dependency graph + business
impact + logs) gives a strong model everything it needs to diagnose and
remediate each incident without requiring exploration or check_logs steps.

---

## API Reference

| Method | Path | Description |
|---|---|---|
| GET | `/` | API overview |
| GET | `/health` | Liveness probe |
| GET | `/schema` | Full action / observation space schema |
| GET | `/tasks` | Task catalogue |
| POST | `/reset` | Start a new episode — body: `{"task_id": "..."}` |
| POST | `/step` | Execute one action — body: `{"type": "...", "target": "...", "params": {}}` |
| GET | `/state` | Read current observation (no state change) |
| GET | `/metrics` | Aggregate episode statistics |
| GET | `/docs` | Swagger UI |

---

## File Structure

```
.
├── env.py            # IncidentEnv — reset / step / state
├── tasks.py          # TaskDefinition — 3 tasks with full telemetry
├── grader.py         # Deterministic reward / scoring
├── app.py            # FastAPI server — all HTTP endpoints
├── inference.py      # Baseline agent — OpenAI client + HF_TOKEN
├── openenv.yaml      # OpenEnv metadata / schema
├── Dockerfile        # Container (HuggingFace Spaces compatible)
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

---

## License

MIT

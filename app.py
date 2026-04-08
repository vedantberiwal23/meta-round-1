'""
app.py — FastAPI server for the Production Incident Response Simulator.

Endpoints:
  GET  /              — API overview
  GET  /health        — liveness probe (required by HuggingFace Spaces)
  GET  /schema        — action / observation space schema
  GET  /tasks         — catalogue of available tasks
  POST /reset         — start a new episode
  POST /step          — execute one action
  GET  /state         — read current observation (no state change)
  GET  /metrics       — episode statistics (for monitoring / dashboards)
"""

import threading
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import time

from env import (
    IncidentEnv, Action, Observation, StepResult,
    VALID_ACTION_TYPES, VALID_TARGETS,
)
from tasks import TASKS

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Production Incident Response Simulator",
    description=(
        "An OpenEnv-compliant AI evaluation environment. "
        "Agents act as SRE engineers diagnosing and resolving production incidents "
        "across three tasks of increasing complexity. "
        "Implements reset / step / state with typed Pydantic models and a "
        "deterministic reward function that scores incremental progress."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Single global environment instance (one active episode at a time)
ENV = IncidentEnv()
ENV_LOCK = threading.Lock()   # prevents state corruption on concurrent requests

# Episode-level stats (for /metrics)
_stats = {
    "total_episodes": 0,
    "episodes_by_task": {"easy": 0, "medium": 0, "hard": 0},
    "total_steps": 0,
    "solved_count": 0,
    "server_start": time.time(),
}


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class ResetRequest(BaseModel):
    task_id: Optional[str] = "easy"   # ← FIXED: fully optional with default


class TaskInfo(BaseModel):
    id: str
    name: str
    difficulty: str
    scenario: str
    max_steps: int
    required_actions: int
    hint: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    """API overview — useful for browsers and automated validators."""
    return {
        "name":        "Production Incident Response Simulator",
        "version":     "2.0.0",
        "openenv":     True,
        "description": (
            "SRE agent evaluation environment with 3 real-world incident tasks "
            "(easy, medium, hard). Implements the OpenEnv interface: "
            "reset / step / state with typed observations, actions, and rewards."
        ),
        "endpoints": {
            "GET  /health":    "Liveness probe",
            "GET  /metadata":  "Environment name, description, version",
            "GET  /schema":    "Action / observation / state space definitions",
            "GET  /tasks":     "Available task catalogue",
            "POST /reset":     "Start a new episode — body: {task_id}",
            "POST /step":      "Execute one action — body: {type, target, params}",
            "GET  /state":     "Read current observation without acting",
            "POST /mcp":       "JSON-RPC 2.0 MCP endpoint",
            "GET  /metrics":   "Episode statistics",
            "GET  /docs":      "Swagger UI",
        },
        "quickstart": (
            "curl -X POST http://localhost:7860/reset "
            "-H 'Content-Type: application/json' "
            "-d '{\"task_id\": \"easy\"}'"
        ),
    }


@app.get("/health")
def health():
    """
    Liveness probe — always returns 200 when the server is running.
    Required by HuggingFace Spaces container health checks.
    """
    return {"status": "healthy", "service": "incident-response-simulator", "version": "2.0.0"}


@app.get("/metadata")
def metadata():
    """
    Environment metadata — required by the OpenEnv runtime validation contract.
    Returns name, description, and version of this environment.
    """
    return {
        "name": "Production Incident Response Simulator",
        "description": (
            "SRE agent evaluation environment with 3 real-world incident tasks "
            "(easy, medium, hard). Agents diagnose and resolve production incidents "
            "via reset / step / state with typed observations, actions, and rewards."
        ),
        "version": "2.0.0",
        "tasks": list(TASKS.keys()),
    }


@app.get("/schema")
def schema():
    """
    Return the full action-space and observation-space schema.
    Useful for agents that want to introspect available actions before acting.
    Keys follow the OpenEnv runtime contract: action / observation / state.
    """
    action_schema = {
        "type": "discrete",
        "description": "Choose one action per step.",
        "fields": {
            "type":   {"type": "str", "enum": VALID_ACTION_TYPES},
            "target": {"type": "str", "enum": VALID_TARGETS},
            "params": {"type": "dict", "description": "Optional extra parameters (usually empty)"},
        },
        "actions": {
            "restart_service": "Restart a crashed or OOMKilled container.",
            "scale_service":   "Horizontally scale an overloaded service (add replicas).",
            "fix_query":       "Apply a database index or rewrite a slow query.",
            "clear_cache":     "Flush stale cache to stop retry storms.",
            "rollback":        "Roll back to the last known-good deploy version.",
            "check_logs":      "Read-only diagnostic. Does not change state; costs a step.",
        },
    }
    observation_schema = {
        "type": "object",
        "fields": {
            "task_id":         "str — 'easy' | 'medium' | 'hard'",
            "task_name":       "str — human-readable task title",
            "difficulty":      "str — 'easy' | 'medium' | 'hard'",
            "scenario":        "str — incident description and objective",
            "hint":            "str — optional hint for human-readability",
            "services":        "List[ServiceStatus] — live telemetry for each service",
            "alerts":          "List[Alert] — active alerts derived from service state",
            "logs":            "List[str] — rolling window of last 20 log lines",
            "dependencies":    "Dict[str, List[str]] — upstream dependencies per service",
            "business_impact": "BusinessImpact — users affected, revenue at risk, SLA countdown",
            "step_number":     "int — current step (1-indexed)",
            "max_steps":       "int — episode step limit",
            "done":            "bool — True when episode is complete",
            "score_so_far":    "float — current episode score [0.0, 1.0]",
        },
    }
    return {
        # OpenEnv runtime contract keys
        "action":      action_schema,
        "observation": observation_schema,
        "state":       observation_schema,
        # Extended info
        "reward": {
            "range":       [-1.0, 1.0],
            "description": "Per-step incremental reward.",
            "components": {
                "correct_action":   "+0.80/N per correct action in sequence (N = required actions)",
                "completion_bonus": "+0.20 when ALL required actions are completed",
                "wrong_action":     "-0.12 per wrong or unnecessary action",
                "step_overhead":    "-0.02 per step (efficiency incentive)",
            },
            "final_score": "Clamped to [0.0, 1.0]. Perfect agent always scores 1.0.",
        },
    }


@app.post("/mcp")
def mcp(request: dict):
    """
    Minimal JSON-RPC 2.0 endpoint — satisfies the OpenEnv MCP contract.
    Responds to standard JSON-RPC method calls (initialize, tools/list, etc.).
    """
    method  = request.get("method", "")
    req_id  = request.get("id", 1)

    if method == "initialize":
        result = {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {
                "name": "production-incident-response-simulator",
                "version": "2.0.0",
            },
        }
    elif method == "tools/list":
        result = {"tools": []}
    else:
        result = {"status": "ok", "method": method}

    return {"jsonrpc": "2.0", "id": req_id, "result": result}


@app.get("/tasks")
def list_tasks():
    """Return the full task catalogue with metadata and action sequences."""
    return {
        "tasks": [
            TaskInfo(
                id=t.id,
                name=t.name,
                difficulty=t.difficulty,
                scenario=t.scenario_description,
                max_steps=t.max_steps,
                required_actions=len(t.expected_action_sequence),
                hint=t.hint,
            )
            for t in TASKS.values()
        ],
        "total_tasks": len(TASKS),
        "valid_action_types": VALID_ACTION_TYPES,
        "valid_targets": VALID_TARGETS,
    }


@app.post("/reset", response_model=Observation)   # ← FIXED: accepts empty body
def reset(request: Optional[ResetRequest] = None):
    """
    Start a new episode.

    Body (optional):
        {"task_id": "easy" | "medium" | "hard"}

    If no body is sent, defaults to task_id="easy".
    Returns the initial Observation — full telemetry, alerts, logs, and hint.
    """
    if request is None:
        request = ResetRequest()

    if request.task_id not in TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id '{request.task_id}'. Valid options: {list(TASKS.keys())}",
        )
    with ENV_LOCK:
        obs = ENV.reset(request.task_id)
        _stats["total_episodes"] += 1
        if request.task_id in _stats["episodes_by_task"]:
            _stats["episodes_by_task"][request.task_id] += 1

    return obs


@app.post("/step", response_model=StepResult)
def step(action: Action):
    """
    Execute one remediation action.

    Body:
        {
            "type":   "restart_service",
            "target": "payment-service",
            "params": {}
        }

    Returns StepResult with updated Observation, reward signal, done flag,
    and grader breakdown in the info dict.
    """
    try:
        with ENV_LOCK:
            result = ENV.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    with ENV_LOCK:
        _stats["total_steps"] += 1
        if result.done and result.observation.score_so_far >= 0.95:
            _stats["solved_count"] += 1

    return result


@app.get("/state", response_model=Observation)
def state():
    """
    Read the current observation without advancing the episode.
    Returns 400 if no episode has been started (call /reset first).
    """
    try:
        return ENV.state()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/metrics")
def metrics():
    """
    Return aggregate episode statistics.
    Useful for monitoring and reporting baseline performance.
    """
    uptime_s = int(time.time() - _stats["server_start"])
    solve_rate = (
        _stats["solved_count"] / _stats["total_episodes"]
        if _stats["total_episodes"] > 0 else 0.0
    )
    return {
        "uptime_seconds":    uptime_s,
        "total_episodes":    _stats["total_episodes"],
        "episodes_by_task":  _stats["episodes_by_task"],
        "total_steps":       _stats["total_steps"],
        "solved_episodes":   _stats["solved_count"],
        "solve_rate":        round(solve_rate, 4),
        "avg_steps_per_episode": (
            round(_stats["total_steps"] / _stats["total_episodes"], 2)
            if _stats["total_episodes"] > 0 else 0.0
        ),
    }

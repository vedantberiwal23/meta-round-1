"""
tasks.py — Task definitions for the Production Incident Response Simulator.

Three tasks of increasing difficulty, each modelling a real SRE scenario:

  easy   — Single service OOM crash              (1 required action)
  medium — Database overload + API saturation    (2 required actions, order flexible)
  hard   — Cascading auth meltdown               (3 required actions, ORDER CRITICAL)

Each task defines:
  - Realistic initial service telemetry (CPU, memory, error_rate, latency, RPS)
  - Structured log lines that tell the incident story
  - Business impact metadata (users affected, revenue at risk)
  - Service dependency graph (downstream consequences)
  - Expected action sequence + max allowed steps
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class TaskDefinition:
    id: str
    name: str
    difficulty: str                       # "easy" | "medium" | "hard"
    scenario_description: str
    initial_state: Dict[str, Any]         # seed telemetry for the environment
    expected_action_sequence: List[Dict]  # ground-truth fix sequence
    max_steps: int
    hint: str = ""
    order_strict: bool = True             # False → any ordering of required actions scores full credit


# ---------------------------------------------------------------------------
# EASY — Single service OOM crash, one root-cause action required
# ---------------------------------------------------------------------------
TASK_EASY = TaskDefinition(
    id="easy",
    name="Payment Service OOM Crash",
    difficulty="easy",
    scenario_description=(
        "INCIDENT P1 — 03:17 UTC\n"
        "payment-service was OOMKilled (exit 137) after a memory leak introduced "
        "in the v3.8.2 hotfix accumulated over 4 hours. The Kubernetes pod is in "
        "CrashLoopBackOff. All /checkout requests are returning 503 to ~120,000 "
        "active users. api-gateway is healthy but routing nowhere for payment calls. "
        "Objective: restore payment-service to bring checkout back online."
    ),
    initial_state={
        "services": {
            "payment-service": {
                "status": "down",
                "cpu": 0.0,
                "memory": 99.8,
                "error_rate": 1.0,
                "latency_ms": 0,
                "request_rate": 0,
            },
            "api-gateway": {
                "status": "degraded",
                "cpu": 48.0,
                "memory": 58.0,
                "error_rate": 0.38,
                "latency_ms": 920,
                "request_rate": 310,
            },
            "frontend": {
                "status": "healthy",
                "cpu": 32.0,
                "memory": 41.0,
                "error_rate": 0.04,
                "latency_ms": 195,
                "request_rate": 420,
            },
            "user-db": {
                "status": "healthy",
                "cpu": 24.0,
                "memory": 49.0,
                "error_rate": 0.01,
                "latency_ms": 9,
                "request_rate": 210,
            },
            "auth-service": {
                "status": "healthy",
                "cpu": 19.0,
                "memory": 36.0,
                "error_rate": 0.01,
                "latency_ms": 72,
                "request_rate": 310,
            },
        },
        "logs": [
            "[03:13:05] CRITICAL  payment-service: java.lang.OutOfMemoryError — heap exhausted",
            "[03:13:06] CRITICAL  payment-service: OOMKilled — container exit code 137",
            "[03:13:07] CRITICAL  payment-service: Pod entered CrashLoopBackOff (restart #7)",
            "[03:13:08] ERROR     api-gateway: upstream payment-service unreachable (connect timeout)",
            "[03:13:09] ERROR     api-gateway: 38% of POST /checkout requests returning 503",
            "[03:13:10] WARNING   api-gateway: circuit breaker OPEN for payment-service route",
            "[03:15:00] INFO      user-db: all 48 connections healthy, query latency normal",
            "[03:15:01] INFO      auth-service: token validation operating normally",
            "[03:15:02] INFO      frontend: serving static assets normally; checkout JS throwing client-side 503",
            "[03:17:30] INFO      k8s: payment-service last successful start was 4h 02m ago (v3.8.2)",
        ],
        "business_impact": {
            "users_affected": 120_000,
            "revenue_at_risk_per_min": 4_800,
            "sla_breach_in_min": 3,
        },
        "dependencies": {
            "payment-service": [],
            "api-gateway": ["payment-service", "auth-service"],
            "frontend": ["api-gateway"],
            "user-db": [],
            "auth-service": ["user-db"],
        },
    },
    expected_action_sequence=[
        {"type": "restart_service", "target": "payment-service", "params": {}},
    ],
    max_steps=5,
    hint=(
        "One service is completely DOWN (exit code 137 = OOMKilled). "
        "Restarting it will restore the memory limits imposed by Kubernetes. "
        "The api-gateway circuit breaker will close once the upstream is healthy again."
    ),
)


# ---------------------------------------------------------------------------
# MEDIUM — Slow DB query + saturated API gateway, two-step fix
# ---------------------------------------------------------------------------
TASK_MEDIUM = TaskDefinition(
    id="medium",
    name="Black Friday DB Overload",
    difficulty="medium",
    scenario_description=(
        "INCIDENT P1 — 11:58 UTC (Black Friday)\n"
        "Deploy v4.1.0 introduced a new analytics query that performs a full-table "
        "scan on the orders table (38M rows) without using the composite index. "
        "Under Black Friday load (600 rps), user-db CPU is pinned at 92% and "
        "average query latency is 3.8 s. The api-gateway request queue has grown "
        "to 1,400 pending requests and its CPU has hit 87%, causing frontend "
        "timeouts. Two actions required — in either order:\n"
        "  1. Scale the api-gateway to drain the backlogged queue.\n"
        "  2. Fix the slow query to eliminate the root cause in user-db."
    ),
    initial_state={
        "services": {
            "payment-service": {
                "status": "healthy",
                "cpu": 31.0,
                "memory": 46.0,
                "error_rate": 0.02,
                "latency_ms": 145,
                "request_rate": 200,
            },
            "api-gateway": {
                "status": "degraded",
                "cpu": 87.0,
                "memory": 78.0,
                "error_rate": 0.18,
                "latency_ms": 4_100,
                "request_rate": 600,
            },
            "frontend": {
                "status": "degraded",
                "cpu": 41.0,
                "memory": 52.0,
                "error_rate": 0.13,
                "latency_ms": 4_800,
                "request_rate": 520,
            },
            "user-db": {
                "status": "degraded",
                "cpu": 92.0,
                "memory": 89.0,
                "error_rate": 0.08,
                "latency_ms": 3_800,
                "request_rate": 600,
            },
            "auth-service": {
                "status": "healthy",
                "cpu": 23.0,
                "memory": 39.0,
                "error_rate": 0.01,
                "latency_ms": 78,
                "request_rate": 360,
            },
        },
        "logs": [
            "[11:46:00] INFO      deploy: v4.1.0 pushed to production (includes new orders analytics feature)",
            "[11:58:12] ERROR     user-db: slow query alert — avg latency 3810 ms (SLO: 200 ms)",
            "[11:58:12] ERROR     user-db: query plan shows FULL TABLE SCAN on orders (38M rows, no index)",
            "[11:58:12] ERROR     user-db: query: SELECT user_id, SUM(total) FROM orders GROUP BY DATE(created_at)",
            "[11:58:15] WARNING   api-gateway: CPU 87% — request queue depth 1,400 (normal: <50)",
            "[11:58:15] WARNING   api-gateway: p99 latency 4,100 ms (SLO: 500 ms)",
            "[11:58:16] WARNING   frontend: upstream timeout rate 13% — users seeing loading spinners",
            "[11:58:20] INFO      k8s: api-gateway running 2 replicas (auto-scale threshold: 80% CPU)",
            "[11:58:21] INFO      user-db: connection pool 98% utilised (max 200 connections)",
            "[11:58:30] INFO      auth-service: operating normally — not involved in slow query path",
        ],
        "business_impact": {
            "users_affected": 280_000,
            "revenue_at_risk_per_min": 22_000,
            "sla_breach_in_min": 1,
        },
        "dependencies": {
            "payment-service": ["user-db", "auth-service"],
            "api-gateway": ["user-db", "auth-service", "payment-service"],
            "frontend": ["api-gateway"],
            "user-db": [],
            "auth-service": ["user-db"],
        },
    },
    expected_action_sequence=[
        {"type": "scale_service",  "target": "api-gateway", "params": {}},
        {"type": "fix_query",      "target": "user-db",     "params": {}},
    ],
    max_steps=8,
    hint=(
        "Two problems, two fixes. The api-gateway is saturated by the backlog — "
        "scale it first to stop timeouts while you work on the root cause. "
        "Then fix the unindexed full-table scan in user-db that was introduced by v4.1.0."
    ),
    order_strict=False,   # either action first is acceptable; both must be done
)


# ---------------------------------------------------------------------------
# HARD — Cascading auth meltdown, three ordered steps
# ---------------------------------------------------------------------------
TASK_HARD = TaskDefinition(
    id="hard",
    name="Auth Service Cascade — Total Outage",
    difficulty="hard",
    scenario_description=(
        "INCIDENT P0 — 22:43 UTC\n"
        "Deploy v5.0.0 shipped a Helm chart with an empty JWT_SECRET env var "
        "(secret store binding was renamed but not updated). auth-service panicked "
        "on startup and entered CrashLoopBackOff. Because every API request passes "
        "through auth middleware, the api-gateway began returning 502 Bad Gateway "
        "to ALL users. The frontend's session cache contains 800 rps of clients "
        "retrying dead auth endpoints, amplifying load and preventing recovery.\n\n"
        "This is a cascading failure with strict dependency ordering:\n"
        "  (1) Restart auth-service — reloads the corrected JWT_SECRET from the "
        "      secret store (ops team patched it 5 min ago).\n"
        "  (2) Scale api-gateway — drain the 2,400-request backlog that built up "
        "      while auth was down.\n"
        "  (3) Clear frontend cache — terminate the retry storm hitting dead "
        "      endpoints so real user traffic can flow again.\n"
        "Order is mandatory: fixing step 2 or 3 first has no effect."
    ),
    initial_state={
        "services": {
            "auth-service": {
                "status": "down",
                "cpu": 0.0,
                "memory": 99.2,
                "error_rate": 1.0,
                "latency_ms": 0,
                "request_rate": 0,
            },
            "api-gateway": {
                "status": "down",
                "cpu": 96.0,
                "memory": 91.0,
                "error_rate": 0.98,
                "latency_ms": 30_000,
                "request_rate": 0,
            },
            "frontend": {
                "status": "degraded",
                "cpu": 73.0,
                "memory": 82.0,
                "error_rate": 0.91,
                "latency_ms": 28_000,
                "request_rate": 800,
            },
            "user-db": {
                "status": "healthy",
                "cpu": 19.0,
                "memory": 31.0,
                "error_rate": 0.01,
                "latency_ms": 8,
                "request_rate": 40,
            },
            "payment-service": {
                "status": "healthy",
                "cpu": 11.0,
                "memory": 26.0,
                "error_rate": 0.0,
                "latency_ms": 48,
                "request_rate": 0,
            },
        },
        "logs": [
            "[22:43:01] CRITICAL  auth-service: panic — JWT_SECRET env var is empty string (Helm chart v5.0.0)",
            "[22:43:01] CRITICAL  auth-service: failed to initialise token validator — exiting (code 1)",
            "[22:43:02] CRITICAL  auth-service: Pod CrashLoopBackOff (restart #12, back-off 5m)",
            "[22:43:03] CRITICAL  api-gateway: auth middleware unreachable — all /api/* returning 502 Bad Gateway",
            "[22:43:04] ERROR     api-gateway: request queue depth 2,400 — CPU 96% — unable to process",
            "[22:43:05] ERROR     frontend: session token validation failing — client retrying at 800 rps",
            "[22:43:06] ERROR     frontend: stale cache serving expired JWT tokens — amplifying retry storm",
            "[22:43:10] WARNING   api-gateway: health checks failing — k8s marking pod NotReady",
            "[22:47:00] INFO      ops: Helm secret binding corrected (JWT_SECRET now maps to vault/auth/jwt)",
            "[22:47:01] INFO      ops: restart auth-service to pick up corrected secret binding",
            "[22:47:02] INFO      payment-service: idle — no authenticated requests reaching it",
            "[22:47:03] INFO      user-db: idle — minimal read traffic only",
        ],
        "business_impact": {
            "users_affected": 1_400_000,
            "revenue_at_risk_per_min": 95_000,
            "sla_breach_in_min": 0,
        },
        "dependencies": {
            "auth-service": ["user-db"],
            "api-gateway": ["auth-service"],
            "frontend": ["api-gateway"],
            "payment-service": ["auth-service", "user-db"],
            "user-db": [],
        },
    },
    expected_action_sequence=[
        {"type": "restart_service", "target": "auth-service",  "params": {}},
        {"type": "scale_service",   "target": "api-gateway",   "params": {}},
        {"type": "clear_cache",     "target": "frontend",      "params": {}},
    ],
    max_steps=10,
    hint=(
        "P0 cascading failure. Read the dependency graph carefully:\n"
        "  auth-service → api-gateway → frontend\n"
        "Fix dependencies in upstream-first order:\n"
        "  (1) restart auth-service — ops already patched the secret.\n"
        "  (2) scale api-gateway — drain the 2,400-request backlog.\n"
        "  (3) clear frontend cache — stop the 800-rps retry storm.\n"
        "Attempting step 2 or 3 before step 1 will NOT improve scores."
    ),
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
TASKS: Dict[str, TaskDefinition] = {
    "easy":   TASK_EASY,
    "medium": TASK_MEDIUM,
    "hard":   TASK_HARD,
}


def get_task(task_id: str) -> TaskDefinition:
    if task_id not in TASKS:
        raise ValueError(
            f"Unknown task_id '{task_id}'. "
            f"Available: {list(TASKS.keys())}"
        )
    return TASKS[task_id]

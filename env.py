"""
env.py — Core environment for the Production Incident Response Simulator.

OpenEnv interface:
  reset(task_id)   → Observation          (start fresh episode)
  step(action)     → StepResult           (observation, reward, done, info)
  state()          → Observation          (read-only snapshot)

Design principles:
  • Fully deterministic — same task, same actions → same outcome every time.
  • Cascading state machine — recovering a root-cause service improves dependents.
  • Business impact surfaced — users affected and revenue at risk in every obs.
  • Dependency graph exposed — agents can reason about upstream/downstream effects.
  • Incremental reward — agents receive a signal every step, not just at episode end.
"""

import copy
import time
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field
from tasks import TaskDefinition, get_task
from grader import Grader


# ===========================================================================
# Pydantic Models — the typed contract between environment and agent
# ===========================================================================

class ServiceStatus(BaseModel):
    name: str
    status: str        # "healthy" | "degraded" | "down"
    cpu: float         # percentage 0–100
    memory: float      # percentage 0–100
    error_rate: float  # fraction 0.0–1.0
    latency_ms: int    # p99 latency in milliseconds
    request_rate: int  # current requests per second


class Alert(BaseModel):
    severity: str    # "critical" | "warning" | "info"
    message: str
    service: str
    timestamp: str = Field(default_factory=lambda: time.strftime("%H:%M:%S"))


class BusinessImpact(BaseModel):
    users_affected: int
    revenue_at_risk_per_min: int
    sla_breach_in_min: int


class Observation(BaseModel):
    # Task context
    task_id: str
    task_name: str
    difficulty: str
    scenario: str
    hint: str

    # Live telemetry
    services: List[ServiceStatus]
    logs: List[str]
    alerts: List[Alert]
    dependencies: Dict[str, List[str]]   # service → list of upstream services it needs
    business_impact: BusinessImpact

    # Episode progress
    step_number: int
    max_steps: int
    done: bool
    score_so_far: float


class Action(BaseModel):
    type: str                                  # see VALID_ACTION_TYPES
    target: str                                # service name to act on
    params: Dict[str, Any] = Field(default_factory=dict)


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]


# ===========================================================================
# Action catalogue
# ===========================================================================
VALID_ACTION_TYPES = [
    "restart_service",   # bring a DOWN service back to healthy
    "scale_service",     # add replicas to an overloaded/degraded service
    "fix_query",         # add missing DB index / rewrite slow query
    "clear_cache",       # flush stale frontend/proxy cache
    "rollback",          # revert to last known-good deploy
    "check_logs",        # read-only diagnostic; costs one step
]

VALID_TARGETS = [
    "auth-service",
    "api-gateway",
    "frontend",
    "user-db",
    "payment-service",
]


# ===========================================================================
# Deterministic state-transition table
# key: (action_type, target) → patch applied to that service
# ===========================================================================
ACTION_EFFECTS: Dict[tuple, Dict[str, Any]] = {
    # --- restarts ---
    ("restart_service", "payment-service"): {
        "status": "healthy", "cpu": 27.0, "memory": 48.0,
        "error_rate": 0.02,  "latency_ms": 115, "request_rate": 200,
    },
    ("restart_service", "auth-service"): {
        "status": "healthy", "cpu": 21.0, "memory": 38.0,
        "error_rate": 0.01,  "latency_ms": 75,  "request_rate": 340,
    },
    # --- scale ---
    ("scale_service", "api-gateway"): {
        "status": "healthy", "cpu": 34.0, "memory": 52.0,
        "error_rate": 0.03,  "latency_ms": 175, "request_rate": 600,
    },
    # --- fix query ---
    ("fix_query", "user-db"): {
        "status": "healthy", "cpu": 19.0, "memory": 44.0,
        "error_rate": 0.01,  "latency_ms": 11,  "request_rate": 600,
    },
    # --- clear cache ---
    ("clear_cache", "frontend"): {
        "status": "healthy", "cpu": 27.0, "memory": 41.0,
        "error_rate": 0.02,  "latency_ms": 185, "request_rate": 480,
    },
    # --- rollback (alt fix for auth) ---
    ("rollback", "auth-service"): {
        "status": "healthy", "cpu": 21.0, "memory": 38.0,
        "error_rate": 0.01,  "latency_ms": 75,  "request_rate": 340,
    },
}

# Downstream cascade: when a root-cause service recovers, its direct dependents improve
DOWNSTREAM_EFFECTS: Dict[str, Dict[str, Dict[str, Any]]] = {
    # payment-service recovery → api-gateway partially recovers
    "restart_service::payment-service": {
        "api-gateway": {
            "status": "healthy", "cpu": 41.0, "memory": 57.0,
            "error_rate": 0.03, "latency_ms": 205, "request_rate": 400,
        },
    },
    # auth-service recovery → api-gateway improves but still has backlog
    "restart_service::auth-service": {
        "api-gateway": {
            "status": "degraded", "cpu": 62.0, "memory": 72.0,
            "error_rate": 0.22, "latency_ms": 2_100, "request_rate": 450,
        },
    },
    # scale api-gateway (after auth fix) → drains backlog, frontend improves
    "scale_service::api-gateway": {
        "frontend": {
            "status": "degraded", "cpu": 52.0, "memory": 63.0,
            "error_rate": 0.38, "latency_ms": 7_000, "request_rate": 800,
        },
    },
    # fix user-db query → api-gateway and frontend both recover
    "fix_query::user-db": {
        "api-gateway": {
            "status": "healthy", "cpu": 31.0, "memory": 51.0,
            "error_rate": 0.02, "latency_ms": 148, "request_rate": 600,
        },
        "frontend": {
            "status": "healthy", "cpu": 29.0, "memory": 46.0,
            "error_rate": 0.03, "latency_ms": 195, "request_rate": 500,
        },
    },
}

# Business impact per healthy service (shown in observation)
HEALTHY_IMPACT = BusinessImpact(
    users_affected=0,
    revenue_at_risk_per_min=0,
    sla_breach_in_min=999,
)


def _recovery_log(action_type: str, target: str) -> str:
    msgs = {
        "restart_service": f"[REMEDIATION] {target}: restarted — container healthy, health-checks passing",
        "scale_service":   f"[REMEDIATION] {target}: scaled out — new replicas online, load balanced",
        "fix_query":       f"[REMEDIATION] {target}: index added — query latency reduced from 3.8s to 11ms",
        "clear_cache":     f"[REMEDIATION] {target}: cache flushed — retry storm terminated, fresh content served",
        "rollback":        f"[REMEDIATION] {target}: rolled back to last known-good version — recovering",
        "check_logs":      f"[DIAGNOSTIC]  Agent inspected logs for {target} — no state change",
    }
    return msgs.get(action_type, f"[REMEDIATION] {target}: action '{action_type}' applied")


def _resolve_alerts(svc_state: Dict[str, Dict]) -> List[Dict]:
    """Derive alert list from current service states (deterministic)."""
    alerts = []
    for name, svc in svc_state.items():
        if svc["status"] == "down":
            alerts.append({
                "severity": "critical",
                "message":  f"{name} is DOWN — 100% error rate",
                "service":  name,
            })
        elif svc["status"] == "degraded":
            alerts.append({
                "severity": "warning",
                "message":  (
                    f"{name} DEGRADED — error_rate={svc['error_rate']:.0%}, "
                    f"latency={svc['latency_ms']}ms, cpu={svc['cpu']:.0f}%"
                ),
                "service":  name,
            })
    return alerts


def _compute_business_impact(
    svc_state: Dict[str, Dict],
    initial_impact: Dict[str, Any],
) -> BusinessImpact:
    """
    Business impact scales down as services recover.
    If all services are healthy → impact is zero.
    """
    down_count     = sum(1 for s in svc_state.values() if s["status"] == "down")
    degraded_count = sum(1 for s in svc_state.values() if s["status"] == "degraded")
    total_svcs     = len(svc_state)

    if down_count == 0 and degraded_count == 0:
        return HEALTHY_IMPACT

    severity_ratio = (down_count + 0.5 * degraded_count) / total_svcs
    users  = int(initial_impact.get("users_affected", 0)          * severity_ratio)
    rev    = int(initial_impact.get("revenue_at_risk_per_min", 0) * severity_ratio)
    breach = max(0, int(initial_impact.get("sla_breach_in_min", 999) * (1 - severity_ratio)))

    return BusinessImpact(
        users_affected=users,
        revenue_at_risk_per_min=rev,
        sla_breach_in_min=breach,
    )


# ===========================================================================
# Environment
# ===========================================================================

class IncidentEnv:
    """
    Production Incident Response Simulator — OpenEnv-compliant environment.

    Quickstart:
        env = IncidentEnv()
        obs = env.reset("easy")
        while not obs.done:
            action = agent.decide(obs)
            result = env.step(action)
            obs = result.observation
        print(f"Final score: {obs.score_so_far}")
    """

    def __init__(self):
        self._task:     Optional[TaskDefinition] = None
        self._grader:   Optional[Grader]         = None
        self._svc_state: Dict[str, Dict]          = {}
        self._logs:     List[str]                 = []
        self._step:     int                       = 0
        self._done:     bool                      = False
        self._initial_impact: Dict[str, Any]      = {}

    # ------------------------------------------------------------------
    # reset  — OpenEnv required
    # ------------------------------------------------------------------
    def reset(self, task_id: str) -> Observation:
        """
        Start a new episode for the given task.

        Args:
            task_id: "easy" | "medium" | "hard"

        Returns:
            Initial Observation (all telemetry, alerts, logs).
        """
        self._task   = get_task(task_id)
        self._grader = Grader(self._task)
        self._step   = 0
        self._done   = False

        init = self._task.initial_state
        self._svc_state      = copy.deepcopy(init["services"])
        self._logs           = list(init["logs"])
        self._initial_impact = init.get("business_impact", {})

        return self._build_observation()

    # ------------------------------------------------------------------
    # step  — OpenEnv required
    # ------------------------------------------------------------------
    def step(self, action: Action) -> StepResult:
        """
        Execute one action and return (observation, reward, done, info).

        Args:
            action: Action with type, target, and optional params.

        Returns:
            StepResult containing the updated Observation, reward signal,
            completion flag, and grader breakdown in info.
        """
        if self._task is None:
            raise RuntimeError("Call reset() before step().")
        if self._done:
            raise RuntimeError("Episode already done — call reset() to start a new episode.")

        self._step += 1
        action_dict = {"type": action.type, "target": action.target, "params": action.params}

        # Apply deterministic state transition
        self._apply_action(action.type, action.target)

        # Append human-readable log entry
        self._logs.append(_recovery_log(action.type, action.target))

        # Score this step
        reward = self._grader.record(action_dict)

        # Termination check
        all_fixed   = self._grader.is_complete()
        step_limit  = self._step >= self._task.max_steps
        self._done  = all_fixed or step_limit

        obs = self._build_observation()

        info = {
            "grader_summary": self._grader.summary(),
            "action_valid":   action.type in VALID_ACTION_TYPES,
            "action_recognised": (action.type, action.target) in ACTION_EFFECTS,
            "step": self._step,
        }

        return StepResult(
            observation=obs,
            reward=round(reward, 4),
            done=self._done,
            info=info,
        )

    # ------------------------------------------------------------------
    # state  — OpenEnv required
    # ------------------------------------------------------------------
    def state(self) -> Observation:
        """
        Return the current observation without advancing the episode.
        """
        if self._task is None:
            raise RuntimeError("Call reset() first.")
        return self._build_observation()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _apply_action(self, action_type: str, target: str):
        """Apply state patch for the given action, then propagate cascades."""
        key = (action_type, target)
        patch = ACTION_EFFECTS.get(key)
        if patch and target in self._svc_state:
            self._svc_state[target].update(patch)

        cascade_key = f"{action_type}::{target}"
        for svc_name, svc_patch in DOWNSTREAM_EFFECTS.get(cascade_key, {}).items():
            if svc_name in self._svc_state:
                self._svc_state[svc_name].update(svc_patch)

    def _build_observation(self) -> Observation:
        """Construct a fully-typed Observation from current internal state."""
        services = [
            ServiceStatus(name=name, **vals)
            for name, vals in self._svc_state.items()
        ]
        alerts = [Alert(**a) for a in _resolve_alerts(self._svc_state)]
        impact = _compute_business_impact(self._svc_state, self._initial_impact)
        deps   = self._task.initial_state.get("dependencies", {})

        return Observation(
            task_id=self._task.id,
            task_name=self._task.name,
            difficulty=self._task.difficulty,
            scenario=self._task.scenario_description,
            hint=self._task.hint,
            services=services,
            logs=self._logs[-20:],   # rolling window of last 20 log lines
            alerts=alerts,
            dependencies=deps,
            business_impact=impact,
            step_number=self._step,
            max_steps=self._task.max_steps,
            done=self._done,
            score_so_far=self._grader.final_score() if self._grader else 0.0,
        )

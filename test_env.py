"""
test_env.py — Full verification suite for the Production Incident Response Simulator.
No API key required. Covers environment logic AND all 8 REST API endpoints.

Tests
-----
  Core environment (8 tests)
    1.  easy  perfect run      → 1.0
    2.  medium forward order   → 1.0
    3.  medium reverse order   → 1.0  (order-agnostic)
    4.  hard  perfect run      → 1.0
    5.  partial credit         → >= 0.5333
    6.  wrong action penalty   → < 1.0
    7.  hard wrong-order penalty (out-of-sequence action 1 penalised)
    8.  max_steps exhaustion   → done=True, step count == max_steps

  Error handling (3 tests)
    9.  step() before reset()  → RuntimeError
    10. step() after done      → RuntimeError
    11. invalid task_id        → ValueError

  Observation completeness (1 test)
    12. all required Observation fields present and typed correctly

  Business impact (1 test)
    13. business impact decreases after root-cause service is fixed

  API endpoints via FastAPI TestClient (8 tests)
    14. GET  /          → 200, openenv=True
    15. GET  /health    → 200, status=ok
    16. GET  /schema    → 200, has action_space and observation_space
    17. GET  /tasks     → 200, 3 tasks
    18. POST /reset     → 200, correct task_id
    19. POST /step      → 200, reward and done fields present
    20. GET  /state     → 200, same observation as last step
    21. GET  /metrics   → 200, total_episodes >= 1

Usage:
    python test_env.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import IncidentEnv, Action
from tasks import get_task, TASKS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_passed = 0
_failed = 0


def check(label: str, condition: bool, detail: str = "") -> bool:
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  [PASS] {label}")
    else:
        _failed += 1
        suffix = f"  => {detail}" if detail else ""
        print(f"  [FAIL] {label}{suffix}")
    return condition


def run_sequence(task_id: str, actions: list) -> dict:
    """Run a fixed action sequence; return final episode results."""
    env = IncidentEnv()
    env.reset(task_id)
    result = None
    steps = 0
    for act in actions:
        result = env.step(Action(**act))
        steps += 1
        if result.done:
            break
    return {
        "score":  result.observation.score_so_far,
        "steps":  steps,
        "done":   result.done,
        "grader": result.info.get("grader_summary", {}),
        "reward": result.reward,
        "obs":    result.observation,
    }


def section(title: str):
    print(f"\n  -- {title} --")


# ===========================================================================
# SECTION 1 — Core environment logic
# ===========================================================================

def test_easy_perfect():
    r = run_sequence("easy", [
        {"type": "restart_service", "target": "payment-service", "params": {}},
    ])
    return check(
        f"easy perfect run   score={r['score']:.4f}  steps={r['steps']}",
        r["score"] == 1.0,
        f"expected 1.0, got {r['score']}",
    )


def test_medium_forward_order():
    r = run_sequence("medium", [
        {"type": "scale_service", "target": "api-gateway", "params": {}},
        {"type": "fix_query",     "target": "user-db",     "params": {}},
    ])
    return check(
        f"medium forward order  score={r['score']:.4f}  steps={r['steps']}",
        r["score"] == 1.0,
        f"expected 1.0, got {r['score']}",
    )


def test_medium_reverse_order():
    r = run_sequence("medium", [
        {"type": "fix_query",     "target": "user-db",     "params": {}},
        {"type": "scale_service", "target": "api-gateway", "params": {}},
    ])
    return check(
        f"medium reverse order  score={r['score']:.4f}  steps={r['steps']}  (order-agnostic)",
        r["score"] == 1.0,
        f"expected 1.0, got {r['score']}",
    )


def test_hard_perfect():
    r = run_sequence("hard", [
        {"type": "restart_service", "target": "auth-service",  "params": {}},
        {"type": "scale_service",   "target": "api-gateway",   "params": {}},
        {"type": "clear_cache",     "target": "frontend",      "params": {}},
    ])
    return check(
        f"hard  perfect run   score={r['score']:.4f}  steps={r['steps']}",
        r["score"] == 1.0,
        f"expected 1.0, got {r['score']}",
    )


def test_partial_credit():
    """2 of 3 correct actions, no extras -> 0.5333"""
    r = run_sequence("hard", [
        {"type": "restart_service", "target": "auth-service", "params": {}},
        {"type": "scale_service",   "target": "api-gateway",  "params": {}},
    ])
    expected = round(2 * (0.80 / 3), 4)
    return check(
        f"partial credit (2/3 hard actions)  score={r['score']:.4f} >= {expected:.4f}",
        r["score"] >= expected,
        f"expected >= {expected}, got {r['score']}",
    )


def test_wrong_action_penalty():
    """Wrong first action on easy task must reduce final score below 1.0."""
    r = run_sequence("easy", [
        {"type": "check_logs",      "target": "api-gateway",    "params": {}},  # wrong
        {"type": "restart_service", "target": "payment-service","params": {}},  # correct
    ])
    return check(
        f"wrong action penalty          score={r['score']:.4f} < 1.0",
        r["score"] < 1.0,
        f"expected < 1.0, got {r['score']}",
    )


def test_hard_wrong_order_penalty():
    """
    Hard task is order-strict.  Taking step 2 (scale api-gateway) before
    step 1 (restart auth-service) must produce a lower score than the
    correct 2-action start, because the out-of-order action earns no credit.

    Out-of-order score:  1 correct * (0.80/3) = 0.2667  (only restart matched)
    In-order score:      2 correct * (0.80/3) = 0.5333
    """
    # Out-of-order: scale first (wrong), then restart (correct at pointer 0)
    r_bad = run_sequence("hard", [
        {"type": "scale_service",   "target": "api-gateway",  "params": {}},  # wrong order
        {"type": "restart_service", "target": "auth-service", "params": {}},  # correct
    ])
    # In-order: restart first (correct), then scale (correct)
    r_good = run_sequence("hard", [
        {"type": "restart_service", "target": "auth-service", "params": {}},  # correct
        {"type": "scale_service",   "target": "api-gateway",  "params": {}},  # correct
    ])

    penalised = r_bad["score"] < r_good["score"]
    return check(
        f"hard out-of-order penalised    bad={r_bad['score']:.4f} < good={r_good['score']:.4f}",
        penalised,
        f"expected bad_score < good_score",
    )


def test_max_steps_exhaustion():
    """Episode must end (done=True) when max_steps is reached even without task solved."""
    task = get_task("easy")   # max_steps = 5
    env = IncidentEnv()
    env.reset("easy")
    result = None
    for _ in range(task.max_steps):
        result = env.step(Action(type="check_logs", target="api-gateway", params={}))
    return check(
        f"max_steps exhaustion          done={result.done}  steps={task.max_steps}",
        result.done is True,
        f"expected done=True after {task.max_steps} steps",
    )


# ===========================================================================
# SECTION 2 — Error handling
# ===========================================================================

def test_step_before_reset():
    """step() before reset() must raise RuntimeError."""
    env = IncidentEnv()
    try:
        env.step(Action(type="restart_service", target="payment-service", params={}))
        return check("step() before reset() raises RuntimeError", False, "no exception raised")
    except RuntimeError:
        return check("step() before reset() raises RuntimeError", True)


def test_step_after_done():
    """step() after episode completes must raise RuntimeError."""
    env = IncidentEnv()
    env.reset("easy")
    env.step(Action(type="restart_service", target="payment-service", params={}))
    # episode is done — one more step must raise
    try:
        env.step(Action(type="check_logs", target="api-gateway", params={}))
        return check("step() after done raises RuntimeError", False, "no exception raised")
    except RuntimeError:
        return check("step() after done raises RuntimeError", True)


def test_invalid_task_id():
    """get_task() with unknown id must raise ValueError."""
    try:
        get_task("nonexistent_task")
        return check("invalid task_id raises ValueError", False, "no exception raised")
    except ValueError:
        return check("invalid task_id raises ValueError", True)


# ===========================================================================
# SECTION 3 — Observation completeness
# ===========================================================================

def test_observation_fields():
    """Every required Observation field must be present and correctly typed."""
    env = IncidentEnv()
    obs = env.reset("hard")

    checks_ok = True
    failures = []

    required_fields = {
        "task_id":         str,
        "task_name":       str,
        "difficulty":      str,
        "scenario":        str,
        "hint":            str,
        "services":        list,
        "logs":            list,
        "alerts":          list,
        "dependencies":    dict,
        "business_impact": object,
        "step_number":     int,
        "max_steps":       int,
        "done":            bool,
        "score_so_far":    float,
    }

    obs_dict = obs.model_dump()
    for field, expected_type in required_fields.items():
        val = obs_dict.get(field)
        if val is None and field not in obs_dict:
            checks_ok = False
            failures.append(f"missing field '{field}'")
        elif not isinstance(val, expected_type):
            checks_ok = False
            failures.append(f"'{field}' is {type(val).__name__}, expected {expected_type.__name__}")

    # Check services have all subfields
    if obs_dict.get("services"):
        svc = obs_dict["services"][0]
        for sf in ["name", "status", "cpu", "memory", "error_rate", "latency_ms", "request_rate"]:
            if sf not in svc:
                checks_ok = False
                failures.append(f"service missing subfield '{sf}'")

    # Check business_impact has all subfields
    bi = obs_dict.get("business_impact", {})
    for bf in ["users_affected", "revenue_at_risk_per_min", "sla_breach_in_min"]:
        if bf not in bi:
            checks_ok = False
            failures.append(f"business_impact missing '{bf}'")

    detail = "; ".join(failures) if failures else ""
    return check(
        f"observation completeness       all {len(required_fields)} fields present and typed",
        checks_ok,
        detail,
    )


# ===========================================================================
# SECTION 4 — Business impact
# ===========================================================================

def test_business_impact_decreases():
    """Business impact (users_affected) must decrease after root cause is fixed."""
    env = IncidentEnv()
    obs_before = env.reset("easy")
    users_before = obs_before.business_impact.users_affected

    result = env.step(Action(type="restart_service", target="payment-service", params={}))
    users_after = result.observation.business_impact.users_affected

    return check(
        f"business impact decreases      {users_before:,} -> {users_after:,} users affected",
        users_after < users_before,
        f"expected users_after < users_before ({users_after} >= {users_before})",
    )


# ===========================================================================
# SECTION 5 — API endpoints via FastAPI TestClient
# ===========================================================================

def _get_test_client():
    """Import TestClient lazily so env tests run even if httpx is missing."""
    from fastapi.testclient import TestClient
    from app import app
    return TestClient(app)


def test_api_root():
    client = _get_test_client()
    r = client.get("/")
    ok = r.status_code == 200 and r.json().get("openenv") is True
    return check(
        f"GET  /          status={r.status_code}  openenv=True",
        ok,
        f"body={r.text[:120]}",
    )


def test_api_health():
    client = _get_test_client()
    r = client.get("/health")
    ok = r.status_code == 200 and r.json().get("status") == "healthy"
    return check(
        f"GET  /health    status={r.status_code}  status=ok",
        ok,
        f"body={r.text[:120]}",
    )


def test_api_schema():
    client = _get_test_client()
    r = client.get("/schema")
    body = r.json()
    ok = (
        r.status_code == 200
        and "action" in body
        and "observation" in body
        and "state" in body
        and "reward" in body
    )
    return check(
        f"GET  /schema    status={r.status_code}  has action/observation/state/reward",
        ok,
        f"keys={list(body.keys())}",
    )


def test_api_tasks():
    client = _get_test_client()
    r = client.get("/tasks")
    body = r.json()
    ok = r.status_code == 200 and body.get("total_tasks") == 3
    return check(
        f"GET  /tasks     status={r.status_code}  total_tasks={body.get('total_tasks')}",
        ok,
        f"body={r.text[:120]}",
    )


def test_api_reset():
    client = _get_test_client()
    r = client.post("/reset", json={"task_id": "easy"})
    body = r.json()
    ok = (
        r.status_code == 200
        and body.get("task_id") == "easy"
        and body.get("difficulty") == "easy"
        and isinstance(body.get("services"), list)
    )
    return check(
        f"POST /reset     status={r.status_code}  task_id=easy  has services",
        ok,
        f"body keys={list(body.keys())[:6]}",
    )


def test_api_step():
    client = _get_test_client()
    client.post("/reset", json={"task_id": "easy"})
    r = client.post("/step", json={
        "type":   "restart_service",
        "target": "payment-service",
        "params": {},
    })
    body = r.json()
    ok = (
        r.status_code == 200
        and "reward" in body
        and "done" in body
        and "observation" in body
        and body.get("done") is True       # easy task: 1 action = done
        and body.get("reward", 0) > 0      # correct action -> positive reward
    )
    return check(
        f"POST /step      status={r.status_code}  done=True  reward={body.get('reward', '?')}",
        ok,
        f"body keys={list(body.keys())}",
    )


def test_api_state():
    client = _get_test_client()
    client.post("/reset", json={"task_id": "medium"})
    r = client.get("/state")
    body = r.json()
    ok = (
        r.status_code == 200
        and body.get("task_id") == "medium"
        and isinstance(body.get("services"), list)
    )
    return check(
        f"GET  /state     status={r.status_code}  task_id=medium",
        ok,
        f"body={r.text[:120]}",
    )


def test_api_metrics():
    client = _get_test_client()
    # Ensure at least one episode has run
    client.post("/reset", json={"task_id": "easy"})
    r = client.get("/metrics")
    body = r.json()
    ok = (
        r.status_code == 200
        and "total_episodes" in body
        and body.get("total_episodes", 0) >= 1
        and "solve_rate" in body
    )
    return check(
        f"GET  /metrics   status={r.status_code}  total_episodes={body.get('total_episodes', '?')}",
        ok,
        f"body={r.text[:120]}",
    )


# ===========================================================================
# Main
# ===========================================================================

def main():
    print()
    print("=" * 68)
    print("  Production Incident Response Simulator — Full Test Suite")
    print("=" * 68)

    section("Core environment logic")
    test_easy_perfect()
    test_medium_forward_order()
    test_medium_reverse_order()
    test_hard_perfect()
    test_partial_credit()
    test_wrong_action_penalty()
    test_hard_wrong_order_penalty()
    test_max_steps_exhaustion()

    section("Error handling")
    test_step_before_reset()
    test_step_after_done()
    test_invalid_task_id()

    section("Observation completeness")
    test_observation_fields()

    section("Business impact")
    test_business_impact_decreases()

    section("REST API endpoints  (FastAPI TestClient)")
    try:
        test_api_root()
        test_api_health()
        test_api_schema()
        test_api_tasks()
        test_api_reset()
        test_api_step()
        test_api_state()
        test_api_metrics()
    except ImportError as e:
        print(f"  [SKIP] TestClient not available: {e}")

    total = _passed + _failed
    print()
    print("=" * 68)
    if _failed == 0:
        print(f"  ALL TESTS PASSED  ({_passed}/{total})")
    else:
        print(f"  FAILED: {_passed}/{total} passed  ({_failed} FAILED)")
    print("=" * 68)
    print()

    sys.exit(0 if _failed == 0 else 1)


if __name__ == "__main__":
    main()

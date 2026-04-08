"""
inference.py — Baseline agent for the Production Incident Response Simulator.

Uses the OpenAI-compatible Hugging Face Inference API so any HF-hosted model
can be evaluated without changing a single line of agent code.

Configuration (environment variables — matches OpenEnv sample spec exactly):
  API_BASE_URL      — LLM inference endpoint base URL
                      (default: "<your-active-endpoint>")
  MODEL_NAME        — Model ID to use for inference
                      (default: "<your-active-model>")
  HF_TOKEN          — HuggingFace access token (required, no default)
  LOCAL_IMAGE_NAME  — Optional: Docker image name when using from_docker_image()
  ENV_URL           — Simulator REST API base URL (default: http://localhost:7860)

Output format — structured stdout logs (one line per event):
  [START] task_id=<id> task_name=<name> difficulty=<level>
  [STEP]  step=<N> action=<type>(<target>) reward=<R> score=<S> done=<bool>
  [END]   task_id=<id> score=<S> steps=<N> status=<SOLVED|PARTIAL|FAILED>

Usage:
  export HF_TOKEN="hf_..."
  export API_BASE_URL="https://api-inference.huggingface.co/v1"
  export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
  python inference.py
"""

import os
import json
import sys
import httpx
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration — variable names match the OpenEnv sample spec exactly
# ---------------------------------------------------------------------------
API_BASE_URL     = os.getenv("API_BASE_URL", "<your-active-endpoint>")
MODEL_NAME       = os.getenv("MODEL_NAME",   "<your-active-model>")
HF_TOKEN         = os.getenv("HF_TOKEN")                     # required — no default
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")              # optional
ENV_URL          = os.getenv("ENV_URL", "http://localhost:7860")  # simulator REST API

BENCHMARK = "production-incident-response-simulator"
TASK_IDS = ["easy", "medium", "hard"]
SUCCESS_SCORE_THRESHOLD = 0.95

if not HF_TOKEN:
    print(
        "ERROR: HF_TOKEN environment variable is not set.\n"
        "  export HF_TOKEN='hf_...'\n"
        "  Then re-run: python inference.py",
        file=sys.stderr,
    )
    sys.exit(1)

# All LLM calls use the OpenAI client configured via API_BASE_URL and HF_TOKEN
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

# ---------------------------------------------------------------------------
# System prompt — instructs the model to act as an SRE agent
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
You are an expert Site Reliability Engineer (SRE) diagnosing and resolving production incidents.

You will receive a JSON observation describing the live state of a production system:
services (with CPU, memory, error_rate, latency), alerts, logs, and business impact.

Your job: choose the single best next remediation action and return EXACTLY this JSON:
{
  "type": "<action_type>",
  "target": "<service_name>",
  "params": {}
}

No markdown. No explanation. Just the raw JSON object.

## Action types
- restart_service  — restart a crashed or OOMKilled pod
- scale_service    — add replicas to an overloaded service
- fix_query        — add a database index / rewrite a slow query
- clear_cache      — flush stale frontend/proxy cache to stop retry storms
- rollback         — roll back to the last known-good deploy
- check_logs       — read-only diagnostic (no state change; wastes a step — avoid unless stuck)

## Valid targets
auth-service | api-gateway | frontend | user-db | payment-service

## Decision strategy
1. Check CRITICAL alerts first — they point to the root cause.
2. Read the dependency graph: fix the upstream service that other services depend on.
3. Fix the ROOT CAUSE first, then downstream symptoms.
4. For cascading failures: auth -> api-gateway -> frontend (upstream to downstream).
5. Be efficient — unnecessary steps reduce your score.
""".strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def obs_to_prompt(obs: dict) -> str:
    """Convert an observation dict to a structured text prompt for the LLM."""
    impact = obs.get("business_impact", {})
    lines = [
        f"# INCIDENT: {obs['task_name']} [{obs['difficulty'].upper()}]",
        f"Step {obs['step_number']}/{obs['max_steps']}  |  Score so far: {obs['score_so_far']:.4f}",
        "",
        "## Scenario",
        obs["scenario"],
        "",
        "## Business Impact",
        f"  Users affected         : {impact.get('users_affected', 0):,}",
        f"  Revenue at risk/min    : ${impact.get('revenue_at_risk_per_min', 0):,}",
        f"  SLA breach countdown   : {impact.get('sla_breach_in_min', 999)} min",
        "",
        "## Active Alerts",
    ]
    alerts = obs.get("alerts", [])
    if alerts:
        for a in alerts:
            lines.append(f"  [{a['severity'].upper():8s}] {a['service']}: {a['message']}")
    else:
        lines.append("  (none -- all services healthy)")

    lines.append("\n## Service Telemetry")
    lines.append(
        f"  {'SERVICE':<22} {'STATUS':<10} {'CPU':>6} {'MEM':>6} "
        f"{'ERR_RATE':>9} {'LATENCY':>10} {'RPS':>6}"
    )
    lines.append("  " + "-" * 74)
    for svc in obs["services"]:
        lines.append(
            f"  {svc['name']:<22} {svc['status']:<10} "
            f"{svc['cpu']:>5.1f}% {svc['memory']:>5.1f}% "
            f"{svc['error_rate']:>8.0%} {svc['latency_ms']:>9}ms "
            f"{svc['request_rate']:>5} rps"
        )

    deps = obs.get("dependencies", {})
    if deps:
        lines.append("\n## Service Dependencies  (service: needs these upstream services)")
        for svc, upstreams in deps.items():
            if upstreams:
                lines.append(f"  {svc}: <- {', '.join(upstreams)}")

    lines.append("\n## Recent Logs  (newest last)")
    for log in obs.get("logs", [])[-10:]:
        lines.append(f"  {log}")

    lines.append("\n## Hint")
    lines.append(f"  {obs.get('hint', '')}")

    lines.append(
        "\nReturn ONLY a valid JSON object with keys: type, target, params. "
        "No markdown fences. No explanations."
    )
    return "\n".join(lines)


def call_model(messages: list) -> str:
    """Call the LLM via OpenAI-compatible endpoint (API_BASE_URL)."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        max_tokens=256,
        temperature=0.05,   # near-deterministic -- SRE decisions should be consistent
        seed=42,            # reproducibility
    )
    return response.choices[0].message.content.strip()


def parse_action(raw: str) -> dict:
    """Extract JSON action from model response (handles accidental markdown fences)."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start != -1 and end > start:
        return json.loads(text[start:end])
    raise ValueError(f"No JSON object found in model response: {raw!r}")


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------
def run_task(task_id: str) -> dict:
    """Run one full episode; return result dict with structured stdout logs."""

    # ---- reset ----
    reset_resp = httpx.post(
        f"{ENV_URL}/reset",
        json={"task_id": task_id},
        timeout=30.0,
    )
    reset_resp.raise_for_status()
    obs = reset_resp.json()

    # [START] — mandatory format: task=<task> env=<benchmark> model=<model>
    print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    rewards  = []          # per-step reward list for [END] line
    steps    = 0
    done     = obs.get("done", False)
    messages = []          # conversation history for multi-turn context

    while not done:
        # Build prompt from current observation
        user_msg = obs_to_prompt(obs)
        messages.append({"role": "user", "content": user_msg})

        # Get model decision via OpenAI client (API_BASE_URL + MODEL_NAME)
        error_str = None
        try:
            raw    = call_model(messages)
            action = parse_action(raw)
            messages.append({"role": "assistant", "content": raw})
        except Exception as e:
            error_str = str(e)
            print(f"[ERROR] step={steps + 1:02d} parse/API error: {e}", file=sys.stderr)
            # Fallback: safe read-only diagnostic action
            action = {"type": "check_logs", "target": "api-gateway", "params": {}}

        # Send action to environment
        step_resp = httpx.post(
            f"{ENV_URL}/step",
            json=action,
            timeout=30.0,
        )
        step_resp.raise_for_status()
        result = step_resp.json()

        reward = result["reward"]
        done   = result["done"]
        obs    = result["observation"]
        steps += 1
        rewards.append(reward)

        action_str = f"{action.get('type', '?')}({action.get('target', '?')})"
        done_val   = str(done).lower()
        error_val  = error_str if error_str else "null"

        # [STEP] — mandatory format: step reward done error (reward=2dp, done=lowercase)
        print(
            f"[STEP] step={steps} action={action_str} "
            f"reward={reward:.2f} done={done_val} error={error_val}",
            flush=True,
        )

    final_score  = obs.get("score_so_far", 0.001)
    success      = final_score >= SUCCESS_SCORE_THRESHOLD
    success_val  = str(success).lower()
    rewards_str  = ",".join(f"{r:.2f}" for r in rewards)

    # [END] — mandatory format: success steps score rewards
    print(
        f"[END] success={success_val} steps={steps} "
        f"score={final_score:.2f} rewards={rewards_str}",
        flush=True,
    )

    status = "SOLVED" if success else ("PARTIAL" if final_score >= 0.40 else "FAILED")
    return {"task_id": task_id, "score": final_score, "steps": steps, "status": status}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    separator = "=" * 72
    print(separator)
    print("  Production Incident Response Simulator -- Baseline Inference Run")
    print(separator)
    print(f"  Environment : {ENV_URL}")
    print(f"  LLM endpoint: {API_BASE_URL}")
    print(f"  Model       : {MODEL_NAME}")
    print(f"  Tasks       : {', '.join(TASK_IDS)}")
    print(f"  Seed        : 42  (temperature=0.05)")
    if LOCAL_IMAGE_NAME:
        print(f"  Docker image: {LOCAL_IMAGE_NAME}")
    print(separator)
    print()

    results = []
    for task_id in TASK_IDS:
        try:
            result = run_task(task_id)
        except Exception as e:
            print(f"[END] task_id={task_id!r} status=ERROR error={e}", file=sys.stderr)
            result = {"task_id": task_id, "score": 0.001, "steps": 0, "status": "ERROR"}
        results.append(result)
        print()

    # Summary table
    print(separator)
    print("  BASELINE RESULTS")
    print(separator)
    print(f"  {'TASK':<10} {'SCORE':>7}  {'STEPS':>6}  {'STATUS'}")
    print("  " + "-" * 36)
    for r in results:
        marker = "+" if r["status"] == "SOLVED" else ("~" if r["status"] == "PARTIAL" else "x")
        print(
            f"  {marker} {r['task_id']:<9} "
            f"{r['score']:>6.4f}  "
            f"{r['steps']:>5}    "
            f"{r['status']}"
        )
    avg = sum(r["score"] for r in results) / max(len(results), 1)
    print(f"\n  Average score: {avg:.4f}")
    print(separator)


if __name__ == "__main__":
    main()

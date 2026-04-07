"""
inference.py — WebSocket-based inference script for DataSelectEnv

Connects to the environment via WebSocket (/ws) — the required transport
on HF Spaces where HTTP /reset and /step are not accessible.

Usage:
    export HF_TOKEN=hf_...              # or OPENAI_API_KEY=sk-...
    export ENV_HOST=https://your-space.hf.space   # or http://localhost:7860
    export API_BASE_URL=https://api-inference.huggingface.co/v1  # optional
    export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct           # optional
    python inference.py [--host URL]

Runs all 3 tasks sequentially using one WebSocket connection per task,
calls POST /grader after each episode, prints scores and final summary.
Designed to complete in under 20 minutes on 2 vCPU / 8 GB RAM.
"""

import argparse
import asyncio
import json
import os
import sys

import requests
import websockets

# ---------------------------------------------------------------------------
# Config — all overridable via environment variables
# ---------------------------------------------------------------------------

DEFAULT_HOST  = os.environ.get("ENV_HOST",      "http://localhost:7860")
API_BASE_URL  = os.environ.get("API_BASE_URL",  "https://api.openai.com/v1")
MODEL_NAME    = os.environ.get("MODEL_NAME",    "gpt-4o-mini")
SEED          = 42
TASKS         = ["easy", "medium", "hard"]

SYSTEM_PROMPT = """You are an intelligent data curation agent.

Your goal is to select high-quality training data from a noisy pool to improve
a machine learning classifier. At each step you observe the current state and
must choose a data selection strategy.

Observation fields:
- remaining_budget: samples you can still select (integer)
- diversity_score: std-dev of current training set features (higher = more diverse)
- noise_estimate: fraction of noisy (mislabelled) samples remaining in pool
- current_performance: validation score = 1/(1+log_loss), range [0,1]
- samples_available: unlabelled samples remaining in the pool

Respond with ONLY a valid JSON action in this exact format:
{
  "action_type": "select_batch",
  "batch_size": <integer 5-20>,
  "strategy_weights": {
    "uncertainty": <float 0-1>,
    "diversity":   <float 0-1>,
    "random":      <float 0-1>
  }
}

Strategy rules:
- Weights are normalized automatically (no need to sum to 1)
- noise_estimate > 0.2  -> lower uncertainty weight, raise diversity weight
- noise_estimate > 0.4  -> set uncertainty near 0, maximize diversity
- diversity_score < 0.5 -> increase diversity weight
- remaining_budget < 30 -> reduce batch_size to 5
- You may use "action_type": "stop" with batch_size 0 only when
  current_performance > 0.65 AND remaining_budget < 20
- Respond with ONLY the JSON object, no explanation, no markdown fences."""


# ---------------------------------------------------------------------------
# Rule-based fallback (used when LLM is unavailable or errors)
# ---------------------------------------------------------------------------

def rule_based_action(obs: dict) -> dict:
    """Produce a sensible action from the observation without an LLM."""
    noise      = obs.get("noise_estimate", 0.1)
    diversity  = obs.get("diversity_score", 1.0)
    budget     = obs.get("remaining_budget", 100)
    perf       = obs.get("current_performance", 0.5)
    available  = obs.get("samples_available", 100)

    # Batch size: shrink near budget exhaustion
    batch_size = 5 if budget < 30 else 10

    # Weights: penalize uncertainty when noise is high
    if noise > 0.4:
        u, d, r = 0.05, 0.80, 0.15
    elif noise > 0.2:
        u, d, r = 0.20, 0.60, 0.20
    elif diversity < 0.5:
        u, d, r = 0.30, 0.55, 0.15
    else:
        u, d, r = 0.40, 0.40, 0.20

    # Early stop if doing well and nearly out of budget
    if perf > 0.65 and budget < 20 and available > 0:
        return {"action_type": "stop", "batch_size": 0,
                "strategy_weights": {"uncertainty": u, "diversity": d, "random": r}}

    return {
        "action_type": "select_batch",
        "batch_size": batch_size,
        "strategy_weights": {"uncertainty": u, "diversity": d, "random": r},
    }


# ---------------------------------------------------------------------------
# LLM helper — uses requests directly (no openai SDK dependency)
# ---------------------------------------------------------------------------

def query_llm(api_key: str | None, obs: dict) -> dict:
    """
    Call the LLM via plain HTTP (OpenAI-compatible chat/completions endpoint).
    Returns a parsed action dict. Raises on any error so the caller can
    fall back to rule_based_action.
    """
    if not api_key:
        raise ValueError("No API key available")

    base_url = (API_BASE_URL or "https://api.openai.com/v1").rstrip("/")
    url = f"{base_url}/chat/completions"

    user_msg = (
        f"Current observation:\n{json.dumps(obs, indent=2)}\n\n"
        "What action do you take?"
    )
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        "temperature": 0.0,
        "max_tokens": 200,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    resp = requests.post(url, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"].strip()

    # Strip markdown fences if model wraps JSON
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    action = json.loads(raw.strip())
    assert "action_type" in action
    assert "batch_size"  in action
    assert "strategy_weights" in action
    return action


# ---------------------------------------------------------------------------
# WebSocket episode runner
# ---------------------------------------------------------------------------

def http_base(host: str) -> str:
    return host.rstrip("/")


def ws_url(host: str) -> str:
    base = http_base(host)
    if base.startswith("https://"):
        return "wss://" + base[len("https://"):] + "/ws"
    if base.startswith("http://"):
        return "ws://" + base[len("http://"):] + "/ws"
    return base + "/ws"


async def run_task_ws(host: str, api_key: str | None, task_id: str) -> dict:
    """Run one full episode for task_id over a WebSocket. Returns grader result."""
    print(f"\n{'='*52}")
    print(f"  Task: {task_id.upper()}")
    print(f"{'='*52}")

    url = ws_url(host)
    print(f"  Connecting to {url} ...")

    async with websockets.connect(url, open_timeout=30, ping_interval=20) as ws:

        # ── reset ────────────────────────────────────────────────────────
        await ws.send(json.dumps({
            "type": "reset",
            "data": {"task_id": task_id, "seed": SEED},
        }))
        resp = json.loads(await ws.recv())
        if resp["type"] == "error":
            raise RuntimeError(f"reset error: {resp['data']['message']}")

        episode_id = resp["data"]["episode_id"]
        obs        = resp["data"]["observation"]
        print(f"  Episode ID: {episode_id}")
        print(f"  Initial obs: {obs}")

        step         = 0
        total_reward = 0.0
        done         = False

        # ── step loop ────────────────────────────────────────────────────
        while not done:
            step += 1

            # Try LLM; fall back to rule-based on any failure
            try:
                action = query_llm(api_key, obs)
            except Exception as e:
                print(f"  Step {step}: LLM unavailable ({type(e).__name__}), using rule-based")
                action = rule_based_action(obs)

            await ws.send(json.dumps({"type": "step", "data": action}))
            resp = json.loads(await ws.recv())

            if resp["type"] == "error":
                print(f"  Step {step}: server error: {resp['data']['message']}")
                break

            data         = resp["data"]
            obs          = data["observation"]
            raw_reward   = data["reward"]
            reward       = raw_reward["value"] if isinstance(raw_reward, dict) else float(raw_reward)
            done         = data["done"]
            total_reward += reward

            print(
                f"  Step {step:2d} | perf={obs['current_performance']:.4f} "
                f"budget={obs['remaining_budget']:3d} "
                f"reward={reward:+.4f} "
                f"noise_est={obs['noise_estimate']:.3f}"
            )

        # ── close WebSocket cleanly ───────────────────────────────────────
        await ws.send(json.dumps({"type": "close", "data": {}}))
        try:
            await asyncio.wait_for(ws.recv(), timeout=2.0)
        except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
            pass

    print(f"\n  Episode done after {step} steps | total_reward={total_reward:.4f}")
    print(f"  Final performance: {obs['current_performance']:.4f}")

    # ── grade via HTTP ────────────────────────────────────────────────────
    r = requests.post(
        f"{http_base(host)}/grader",
        json={"episode_id": episode_id, "task_id": task_id},
        timeout=15,
    )
    r.raise_for_status()
    grade = r.json()

    print(f"  Score:   {grade['score']:.4f}")
    print(f"  Passed:  {grade['passed']}")
    print(f"  Details: {grade['breakdown']}")

    return {
        "task_id":           task_id,
        "score":             grade["score"],
        "passed":            grade["passed"],
        "breakdown":         grade["breakdown"],
        "steps":             step,
        "total_reward":      round(total_reward, 4),
        "final_performance": obs["current_performance"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def amain(host: str, api_key: str | None) -> None:
    results = {}
    for task_id in TASKS:
        results[task_id] = await run_task_ws(host, api_key, task_id)

    print(f"\n{'='*52}")
    print("  INFERENCE RESULTS SUMMARY")
    print(f"{'='*52}")
    print(f"{'Task':<10} {'Score':<8} {'Passed':<8} {'Final Perf':<12} {'Steps'}")
    print("-" * 52)
    for task_id, r in results.items():
        print(
            f"{task_id:<10} {r['score']:<8.4f} {str(r['passed']):<8} "
            f"{r['final_performance']:<12.4f} {r['steps']}"
        )

    overall = sum(r["score"] for r in results.values()) / len(results)
    print(f"\nOverall mean score: {overall:.4f}")
    print(json.dumps({"results": results, "mean_score": round(overall, 4)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="DataSelectEnv WebSocket inference script")
    parser.add_argument("--host", default=DEFAULT_HOST,
                        help="Environment server base URL (http or https)")
    args = parser.parse_args()

    # API key is optional — rule-based fallback runs without one
    api_key = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
    if api_key:
        print(f"LLM API key found ({len(api_key)} chars); will attempt LLM-guided actions.")
    else:
        print("No API key (HF_TOKEN / OPENAI_API_KEY); running rule-based fallback.")

    # Health check — environment must be reachable
    try:
        r = requests.get(f"{http_base(args.host)}/health", timeout=15)
        r.raise_for_status()
        print(f"Connected to {args.host} — {r.json()}")
    except Exception as e:
        print(f"ERROR: Could not reach environment at {args.host}: {e}")
        sys.exit(1)

    asyncio.run(amain(args.host, api_key))


if __name__ == "__main__":
    main()

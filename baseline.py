"""
baseline.py — Baseline inference script for DataSelectEnv

Uses the OpenAI API client (as required by OpenEnv spec) to run an LLM
agent against all 3 tasks and produce reproducible scores.

Usage:
    export OPENAI_API_KEY=sk-...
    python baseline.py [--host http://localhost:7860]

The agent is given the current observation as a JSON prompt and asked
to return an action. This tests whether an LLM can navigate the
data-curation environment without any fine-tuning.
"""

import argparse
import json
import os
import sys

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_HOST = os.environ.get("ENV_HOST", "http://localhost:7860")
SEED = 42
TASKS = ["easy", "medium", "hard"]

SYSTEM_PROMPT = """You are an intelligent data curation agent.

Your goal is to select high-quality training data from a pool to improve a machine learning model.
At each step you observe the current state and must choose a data selection strategy.

You will receive a JSON observation with these fields:
- remaining_budget: how many samples you can still select
- diversity_score: current diversity of training set (higher = more diverse)
- noise_estimate: estimated fraction of noisy samples remaining in pool
- current_performance: current model validation performance (higher = better)
- samples_available: number of samples left in the pool

You must respond with ONLY a valid JSON action in this exact format:
{
  "action_type": "select_batch",
  "batch_size": <integer between 5 and 20>,
  "strategy_weights": {
    "uncertainty": <float 0-1>,
    "diversity": <float 0-1>,
    "random": <float 0-1>
  }
}

Rules:
- Weights do not need to sum to 1 (they are normalized automatically)
- If noise_estimate is high (>0.2), reduce uncertainty weight and increase diversity weight
- If diversity_score is low (<0.5), increase diversity weight
- If remaining_budget is low (<30), use smaller batch sizes
- You may use "action_type": "stop" to end early if performance is already good
- Respond with ONLY the JSON object, no explanation."""


def query_llm(client: OpenAI, observation: dict) -> dict:
    """Ask the LLM to produce an action given the current observation."""
    user_msg = f"Current observation:\n{json.dumps(observation, indent=2)}\n\nWhat action do you take?"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.0,   # deterministic
        max_tokens=200,
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if the model wraps the JSON
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    return json.loads(raw)


def run_task(host: str, client: OpenAI, task_id: str) -> dict:
    """Run one full episode on a task and return the grader result."""
    print(f"\n{'='*50}")
    print(f"Task: {task_id.upper()}")
    print(f"{'='*50}")

    # Reset
    r = requests.post(f"{host}/reset", json={"task_id": task_id, "seed": SEED})
    r.raise_for_status()
    data       = r.json()
    episode_id = data["episode_id"]
    obs        = data["observation"]
    print(f"Episode ID: {episode_id}")
    print(f"Initial obs: {obs}")

    step = 0
    total_reward = 0.0

    while True:
        step += 1

        # Get action from LLM
        try:
            action = query_llm(client, obs)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Step {step}: LLM returned invalid JSON ({e}), using fallback action")
            action = {
                "action_type": "select_batch",
                "batch_size": 10,
                "strategy_weights": {"uncertainty": 0.4, "diversity": 0.4, "random": 0.2},
            }

        # Execute action
        r = requests.post(f"{host}/step", json={"action": action})
        r.raise_for_status()
        result = r.json()

        obs          = result["observation"]
        reward       = result["reward"]
        done         = result["done"]
        total_reward += reward

        print(f"  Step {step:2d} | perf={obs['current_performance']:.4f} "
              f"budget={obs['remaining_budget']:3d} reward={reward:+.4f} "
              f"noise_est={obs['noise_estimate']:.3f}")

        if done:
            break

    print(f"\nEpisode done after {step} steps | total_reward={total_reward:.4f}")
    print(f"Final performance: {obs['current_performance']:.4f}")

    # Grade
    r = requests.post(f"{host}/grader", json={"episode_id": episode_id, "task_id": task_id})
    r.raise_for_status()
    grade = r.json()

    print(f"Score:   {grade['score']:.4f}")
    print(f"Passed:  {grade['passed']}")
    print(f"Details: {grade['breakdown']}")

    return {
        "task_id":     task_id,
        "score":       grade["score"],
        "passed":      grade["passed"],
        "breakdown":   grade["breakdown"],
        "steps":       step,
        "total_reward": round(total_reward, 4),
        "final_performance": obs["current_performance"],
    }


def main():
    parser = argparse.ArgumentParser(description="DataSelectEnv baseline inference script")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Environment server URL")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # Health check
    try:
        r = requests.get(f"{args.host}/health", timeout=5)
        r.raise_for_status()
        print(f"Connected to {args.host} — {r.json()}")
    except Exception as e:
        print(f"ERROR: Could not reach environment at {args.host}: {e}")
        sys.exit(1)

    # Run all tasks
    results = {}
    for task_id in TASKS:
        results[task_id] = run_task(args.host, client, task_id)

    # Summary
    print(f"\n{'='*50}")
    print("BASELINE RESULTS SUMMARY")
    print(f"{'='*50}")
    print(f"{'Task':<10} {'Score':<8} {'Passed':<8} {'Final Perf':<12} {'Steps'}")
    print("-" * 50)
    for task_id, r in results.items():
        print(f"{task_id:<10} {r['score']:<8.4f} {str(r['passed']):<8} "
              f"{r['final_performance']:<12.4f} {r['steps']}")

    overall = sum(r["score"] for r in results.values()) / len(results)
    print(f"\nOverall mean score: {overall:.4f}")
    print(json.dumps({"baseline_results": results, "mean_score": round(overall, 4)}, indent=2))


if __name__ == "__main__":
    main()
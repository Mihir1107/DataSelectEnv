"""
server.py — OpenEnv-compliant FastAPI server for DataSelectEnv

Endpoints (required by spec):
  POST /reset      — start a new episode
  POST /step       — take one action
  GET  /state      — current episode metadata

Additional endpoints (required by hackathon):
  GET  /tasks      — list all tasks + action schema
  POST /grader     — score a completed episode
  GET  /baseline   — run baseline agent on all 3 tasks
  GET  /health     — liveness check (HF Spaces ping)
"""

import copy
import os
import time
import uuid
from typing import Any, Dict, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env import DataSelectEnv
from models import Action, EnvState, Observation

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="DataSelectEnv",
    description=(
        "OpenEnv RL environment for data curation in ML training. "
        "Agents learn to select high-quality training data under noise, "
        "diversity, and budget constraints."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Base config — tasks override specific fields
# ---------------------------------------------------------------------------

BASE_CFG = {
    "data": {
        "n_samples":    1500,
        "n_features":   20,
        "n_informative": 5,
        "n_redundant":  5,
        "flip_y":       0.1,
    },
    "budget":    300,
    "max_steps": 15,
    "alpha":     0.2,
}

# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASKS = {
    "easy": {
        "task_id":   "easy",
        "difficulty": "easy",
        "description": (
            "Clean dataset (flip_y=0.05), budget=300, max_steps=15. "
            "Agent must reach validation performance > 0.55. "
            "Any reasonable balanced strategy succeeds."
        ),
        "success_criteria": "current_performance > 0.55 at episode end",
        "cfg_overrides": {
            "data":      {"flip_y": 0.05},
            "budget":    300,
            "max_steps": 15,
        },
    },
    "medium": {
        "task_id":   "medium",
        "difficulty": "medium",
        "description": (
            "High noise (flip_y=0.25), budget=150, max_steps=12. "
            "Agent must reach performance > 0.52 while keeping average "
            "noise selection rate below 0.30. Uncertainty-only strategies fail."
        ),
        "success_criteria": "current_performance > 0.52 AND avg noise_ratio < 0.30",
        "cfg_overrides": {
            "data":      {"flip_y": 0.25},
            "budget":    150,
            "max_steps": 12,
        },
    },
    "hard": {
        "task_id":   "hard",
        "difficulty": "hard",
        "description": (
            "High noise (flip_y=0.30), tight budget=100, max_steps=8. "
            "Agent must hit performance > 0.53 efficiently. "
            "Grader scores performance and budget efficiency jointly. "
            "Requires precise noise-aware + diversity-aware strategy."
        ),
        "success_criteria": "performance > 0.53, scored jointly with budget efficiency",
        "cfg_overrides": {
            "data":      {"flip_y": 0.30},
            "budget":    100,
            "max_steps": 8,
        },
    },
}

ACTION_SCHEMA = {
    "action_type": {
        "type": "string",
        "enum": ["select_batch", "stop"],
        "description": "select_batch to select data, stop to end the episode early.",
    },
    "batch_size": {
        "type":        "integer",
        "minimum":     0,
        "description": "Number of samples to select this step.",
    },
    "strategy_weights": {
        "type": "object",
        "properties": {
            "uncertainty": {"type": "number", "minimum": 0.0},
            "diversity":   {"type": "number", "minimum": 0.0},
            "random":      {"type": "number", "minimum": 0.0},
        },
        "description": "Sampling strategy weights — normalized internally.",
    },
}

# ---------------------------------------------------------------------------
# In-memory episode store (single process — fine for HF Spaces)
# ---------------------------------------------------------------------------

class EpisodeStore:
    def __init__(self):
        self.env:          Optional[DataSelectEnv] = None
        self.episode_id:   Optional[str]           = None
        self.task_id:      Optional[str]           = None
        self.step_count:   int                     = 0
        self.done:         bool                    = False
        self.started_at:   Optional[float]         = None
        self.total_reward: float                   = 0.0
        self.noise_ratios: list                    = []
        self.final_obs:    Optional[Observation]   = None
        self._cfg:         Optional[dict]          = None

store = EpisodeStore()

# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    seed:    int            = 42
    task_id: Optional[str] = "easy"

class StepRequest(BaseModel):
    action: Action

class GraderRequest(BaseModel):
    episode_id: str
    task_id:    str

class GraderResponse(BaseModel):
    episode_id: str
    task_id:    str
    score:      float
    breakdown:  Dict[str, Any]
    passed:     bool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_cfg(task_id: str) -> dict:
    cfg = copy.deepcopy(BASE_CFG)
    for key, val in TASKS[task_id]["cfg_overrides"].items():
        if isinstance(val, dict):
            cfg[key].update(val)
        else:
            cfg[key] = val
    return cfg


def _grade(task_id: str, obs: Observation, noise_ratios: list, cfg: dict) -> GraderResponse:
    perf = obs.current_performance

    if task_id == "easy":
        # Single dimension: raw performance
        score = float(np.clip((perf - 0.45) / (0.65 - 0.45), 0.0, 1.0))
        passed = perf > 0.55
        breakdown: Dict[str, Any] = {"performance_score": round(score, 4)}

    elif task_id == "medium":
        avg_noise = float(np.mean(noise_ratios)) if noise_ratios else 1.0
        # Performance sub-score
        perf_score  = float(np.clip((perf - 0.42) / (0.62 - 0.42), 0.0, 1.0))
        # Noise avoidance sub-score: full marks at 0 noise, zero at >=0.30
        noise_score = float(np.clip(1.0 - avg_noise / 0.30, 0.0, 1.0))
        score  = round(0.6 * perf_score + 0.4 * noise_score, 4)
        passed = perf > 0.52 and avg_noise < 0.30
        breakdown = {
            "performance_score": round(perf_score,  4),
            "noise_score":       round(noise_score, 4),
            "avg_noise_ratio":   round(avg_noise,   4),
        }

    else:  # hard
        budget_total = cfg["budget"]
        budget_used  = budget_total - obs.remaining_budget
        perf_score   = float(np.clip((perf - 0.43) / (0.63 - 0.43), 0.0, 1.0))
        # Efficiency: reward finishing with budget left; +0.2 grace so
        # spending most of the budget still gets partial efficiency credit
        efficiency   = float(np.clip(1.0 - budget_used / budget_total + 0.2, 0.0, 1.0))
        score  = round(0.65 * perf_score + 0.35 * efficiency, 4)
        passed = perf > 0.53
        breakdown = {
            "performance_score": round(perf_score, 4),
            "efficiency_score":  round(efficiency, 4),
            "budget_used":       int(budget_used),
        }

    return GraderResponse(
        episode_id=store.episode_id,
        task_id=task_id,
        score=score,
        breakdown=breakdown,
        passed=passed,
    )

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Liveness check — HF Spaces automated ping."""
    return {"status": "ok", "env": "DataSelectEnv", "version": "1.0.0"}


@app.post("/reset")
def reset(req: ResetRequest):
    """
    Start a new episode.

    Body:
      seed    (int, default 42)    — RNG seed for full reproducibility
      task_id (str, default 'easy') — one of: easy | medium | hard
    """
    if req.task_id not in TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id '{req.task_id}'. Valid: {list(TASKS.keys())}",
        )

    cfg = _build_cfg(req.task_id)
    env = DataSelectEnv(cfg, seed=req.seed)
    obs = env.reset()

    store.env          = env
    store.episode_id   = str(uuid.uuid4())
    store.task_id      = req.task_id
    store.step_count   = 0
    store.done         = False
    store.started_at   = time.time()
    store.total_reward = 0.0
    store.noise_ratios = []
    store.final_obs    = obs
    store._cfg         = cfg

    return {
        "episode_id":  store.episode_id,
        "task_id":     req.task_id,
        "observation": obs.model_dump(),
    }


@app.post("/step")
def step(req: StepRequest):
    """
    Execute one action in the current episode.

    Body: { "action": { "action_type": ..., "batch_size": ..., "strategy_weights": {...} } }
    """
    if store.env is None or store.done:
        raise HTTPException(
            status_code=400,
            detail="No active episode. Call POST /reset first.",
        )

    obs, reward, done, info = store.env.step(req.action)

    store.step_count   += 1
    store.done          = done
    store.total_reward += reward
    store.final_obs     = obs

    if "noise_ratio" in info:
        store.noise_ratios.append(info["noise_ratio"])

    return {
        "episode_id":  store.episode_id,
        "step":        store.step_count,
        "observation": obs.model_dump(),
        "reward":      round(float(reward), 6),
        "done":        done,
        "info":        info,
    }


@app.get("/state")
def state():
    """Return current episode metadata (OpenEnv spec requirement)."""
    env_meta = store.env.get_state() if store.env else EnvState(
        step_count=0, remaining_budget=None,
        current_performance=None, pool_size=None, done=False,
    )
    return {
        "episode_id":          store.episode_id,
        "task_id":             store.task_id,
        "step_count":          env_meta.step_count,
        "done":                env_meta.done,
        "remaining_budget":    env_meta.remaining_budget,
        "current_performance": env_meta.current_performance,
        "pool_size":           env_meta.pool_size,
    }


@app.get("/tasks")
def tasks():
    """Return all task definitions and the action schema."""
    return {
        "tasks": [
            {
                "task_id":          t["task_id"],
                "difficulty":       t["difficulty"],
                "description":      t["description"],
                "success_criteria": t["success_criteria"],
                "action_schema":    ACTION_SCHEMA,
            }
            for t in TASKS.values()
        ],
        "action_schema": ACTION_SCHEMA,
    }


@app.post("/grader")
def grader(req: GraderRequest):
    """
    Score the most recently completed episode.

    Body: { "episode_id": "...", "task_id": "easy|medium|hard" }
    episode_id must match the active episode; episode must be done.
    """
    if store.episode_id != req.episode_id:
        raise HTTPException(
            status_code=400,
            detail="episode_id does not match the current episode.",
        )
    if not store.done:
        raise HTTPException(
            status_code=400,
            detail="Episode is not finished. Keep stepping until done=True.",
        )
    if req.task_id not in TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id '{req.task_id}'.",
        )

    return _grade(req.task_id, store.final_obs, store.noise_ratios, store._cfg)


@app.get("/baseline")
def baseline():
    """
    Run a fixed balanced agent on all 3 tasks and return reproducible scores.
    This is the baseline the judges re-run during Phase 2 evaluation.
    Seed is fixed at 42 for reproducibility.
    """
    BASELINE_ACTION = Action(
        action_type="select_batch",
        batch_size=10,
        strategy_weights={"uncertainty": 0.4, "diversity": 0.4, "random": 0.2},
    )

    results = {}

    for task_id in TASKS:
        cfg  = _build_cfg(task_id)
        env  = DataSelectEnv(cfg, seed=42)
        obs  = env.reset()

        noise_ratios: list = []
        total_reward: float = 0.0
        done = False

        while not done:
            obs, reward, done, info = env.step(BASELINE_ACTION)
            total_reward += reward
            if "noise_ratio" in info:
                noise_ratios.append(info["noise_ratio"])

        # Temporarily point store.episode_id so _grade() can fill it
        _saved_id  = store.episode_id
        store.episode_id = str(uuid.uuid4())
        result = _grade(task_id, obs, noise_ratios, cfg)
        store.episode_id = _saved_id

        results[task_id] = {
            "score":               result.score,
            "passed":              result.passed,
            "breakdown":           result.breakdown,
            "total_reward":        round(total_reward, 4),
            "final_performance":   round(float(obs.current_performance), 4),
        }

    return {
        "baseline_agent": "balanced (uncertainty=0.4, diversity=0.4, random=0.2)",
        "seed":           42,
        "results":        results,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
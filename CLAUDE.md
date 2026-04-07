# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the server locally (port 7860)
python3 server.py

# Run tests
python3 tests/test_env.py

# Validate OpenEnv compliance
openenv validate

# Run pre-submission validation (requires Docker Desktop running)
/tmp/validate-submission.sh https://mihir1107-dateselectenv.hf.space .
```

## Architecture

This is an OpenEnv-compliant RL environment simulating active learning / data curation under budget constraints.

**Entry point:** `server.py` — FastAPI app exposing all endpoints. Run directly with `python3 server.py`. The `server/` package (`server/__init__.py` + `server/app.py`) is a shim that exists solely for `openenv validate` compliance (`[project.scripts]` entry point); it loads `server.py` via `importlib` to avoid the naming conflict between `server.py` and `server/`.

**Core environment:** `env.py`
- `DataSelectEnv.reset()` — generates dataset via `make_classification`, injects noise (guaranteed label flip via `1 - y[mask]`), warms up `SGDClassifier` for 10 passes on 100 seed samples
- `DataSelectEnv.step(action)` — selects batch using weighted combination of uncertainty/diversity/random sampling, incrementally trains model via `partial_fit`, computes shaped reward
- Data split: `X[:100]` seed | `X[200:400]` validation | `X[400:]` pool (~1100 samples)
- Performance metric: `1 / (1 + log_loss)` on validation set

**Reward design** (`env.py` step()):
- `gain = perf_delta * 5.0 + mean_distance_from_centroid * 0.05`
- `noise_penalty = (1.0 + flip_y * 2.0) * noise_ratio` — scales with task difficulty
- `reward = gain - 0.01 * batch_size - 0.3 * redundancy - noise_penalty + 0.15`
- Noise trap: entropy of noisy pool samples is boosted by `min(0.1 + flip_y*2.0, 0.55)` so uncertainty sampling is attracted to noisy samples (hurts uncertainty-only strategies on medium/hard)

**Sampling:** `sampling.py` — `sample_uncertainty`, `sample_diversity`, `sample_random`. Weights normalized internally. `min_batch=5` enforced unless action is `stop`.

**Tasks** (defined in `server.py` TASKS dict):
- `easy`: `flip_y=0.05`, budget=300, max_steps=15, grader range [0.55, 0.75]
- `medium`: `flip_y=0.25`, budget=150, max_steps=12, grader: `0.6*perf + 0.4*noise_score`
- `hard`: `flip_y=0.30`, budget=100, max_steps=8, grader: `0.65*perf + 0.35*efficiency`

**Episode persistence:** `/grader` works after a subsequent `/reset` because completed episodes are stored in `_completed` dict keyed by `episode_id`. The `/ws` WebSocket endpoint has fully isolated per-connection state (does not share `store`).

**Deployment:**
- Dockerfile: `CMD ["python", "server.py"]` — uses root `server.py` directly
- HF Space: `https://huggingface.co/spaces/Mihir1107/DateSelectEnv`
- GitHub: `https://github.com/Mihir1107/DataSelectEnv`
- `uvicorn.run(app, ...)` uses the app object directly (not `"server:app"` string) to avoid `server/` package shadowing `server.py`

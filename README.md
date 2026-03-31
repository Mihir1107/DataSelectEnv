---
title: DataSelectEnv
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---

# DataSelectEnv — OpenEnv Environment for Data Curation in ML Training

## Description

DataSelectEnv is an [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compliant reinforcement learning environment for the Meta PyTorch OpenEnv Hackathon. The environment models a core problem in real-world machine learning: given a large pool of candidate training examples — some clean, some mislabelled, all with a cost to acquire — which samples should an agent select to maximise model quality under a fixed labelling budget?

We implement a real incremental training loop using SGDClassifier, where agents must select training data under budget constraints capturing realistic ML dynamics including diminishing returns, redundancy penalties, and noise sensitivity. The agent observes the current classifier's validation performance, an estimate of remaining pool noise, training-set diversity, and budget left, then decides how many samples to select and which sampling strategy to weight. Three difficulty tiers (easy / medium / hard) vary the noise level, budget, and time horizon to test whether agents can adapt their strategy to the environment's constraints.

---

## Observation Space

| Field | Type | Range | Description |
|---|---|---|---|
| `remaining_budget` | int | [0, budget] | Samples remaining in the selection budget |
| `diversity_score` | float | [0, ∞) | Std-dev of current training set features (proxy for diversity) |
| `noise_estimate` | float | [0, 1] | Fraction of noisy samples still remaining in the pool |
| `current_performance` | float | [0, 1] | Validation score: 1 / (1 + log_loss) |
| `samples_available` | int | [0, ~1100] | Number of unlabelled samples left in the pool |

---

## Action Space

| Field | Type | Values | Description |
|---|---|---|---|
| `action_type` | string | `select_batch`, `stop` | Select a batch of data or end the episode early |
| `batch_size` | int | ≥ 0 | Number of samples to select this step |
| `strategy_weights.uncertainty` | float | ≥ 0 | Weight for uncertainty sampling (highest-entropy samples) |
| `strategy_weights.diversity` | float | ≥ 0 | Weight for diversity sampling (farthest from training centroid) |
| `strategy_weights.random` | float | ≥ 0 | Weight for uniform random sampling |

Strategy weights are normalised internally and do not need to sum to 1.

---

## Tasks

| Name | flip_y | Budget | max_steps | Success criteria | Expected random score |
|---|---|---|---|---|---|
| `easy` | 0.05 | 300 | 15 | performance > 0.55 | ~0.60 |
| `medium` | 0.25 | 150 | 12 | performance > 0.52 AND avg noise ratio < 0.30 | ~0.40 |
| `hard` | 0.30 | 100 | 8 | performance > 0.53 (+ budget efficiency) | ~0.30 |

---

## Reward Function

```
gain        = (new_performance - old_performance) * 5.0
            + 0.2 * std(selected_batch)            # diversity bonus
            + 0.2 * (new_performance - old_performance)  # alignment bonus

if redundancy > 0.8:  gain *= 0.5   # redundancy penalty
if new_performance > 0.85: gain *= 0.7  # diminishing-returns cap

noise_penalty = 0.4 * noise_ratio_of_selected_batch

reward = gain
       - 0.01 * batch_size       # budget cost
       - 0.3  * redundancy       # cosine similarity to training centroid
       - noise_penalty
```

---

## Setup

### Local (pip)

```bash
pip install -r requirements.txt
python server.py          # starts on http://localhost:7860
```

### Docker

```bash
docker build -t dataselectenv .
docker run -p 7860:7860 dataselectenv
```

### Run inference (LLM agent)

```bash
export HF_TOKEN=hf_...        # or OPENAI_API_KEY=sk-...
export API_BASE_URL=https://api-inference.huggingface.co/v1   # optional
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct           # optional

python inference.py --host http://localhost:7860
```

### API quick-start

```bash
# Reset an episode
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy", "seed": 42}'

# Take a step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "select_batch", "batch_size": 10,
       "strategy_weights": {"uncertainty": 0.4, "diversity": 0.4, "random": 0.2}}}'

# Run the built-in baseline and get reproducible scores
curl http://localhost:7860/baseline
```

---

## Baseline Scores

Scores below are from the fixed balanced agent (`uncertainty=0.4, diversity=0.4, random=0.2`, seed=42) run via `GET /baseline`.

| Task | Score | Passed | Final performance |
|---|---|---|---|
| easy | 0.7020 | ✅ | 0.6904 |
| medium | 0.6600 | ✅ | 0.6569 |
| hard | 0.4174 | ✅ | 0.6176 |

Scores are from the fixed balanced agent (`uncertainty=0.4, diversity=0.4, random=0.2`, seed=42) via `GET /baseline`.

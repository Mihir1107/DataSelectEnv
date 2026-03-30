# DataSelectEnv — OpenEnv Environment for Data Curation in ML Training

🚀 **Overview**

DataSelectEnv is a reinforcement learning environment that simulates a real-world machine learning workflow:
selecting high-quality training data under constraints.

Agents must decide:
- which data to select
- how much to select
- and which strategy to prioritize

while balancing:
- data quality
- diversity
- noise
- and budget

---

🎯 **Motivation**

In real-world ML systems (e.g., at companies like Meta or Hugging Face), performance is not just about model architecture —
it heavily depends on which data you train on.

This environment models:
- active learning
- data filtering
- noisy dataset handling
- budget-constrained training

---

🧠 **Core Idea**

Instead of training a model once, the agent interacts step-by-step:
1. Observe current model performance and dataset state
2. Select a batch of data using different strategies
3. Incrementally train the model (`partial_fit`)
4. Receive reward based on improvement and data quality

---

⚙️ **Environment Design**

📦 **Dataset**
- Generated using `sklearn.make_classification`
- 1500 samples, 20 features
- Structured clusters with controlled noise
- Split:
  - Seed: 100 samples (initial training)
  - Validation: 200 samples
  - Pool: remaining samples

🔥 **Noise Simulation**
- Some samples have corrupted labels
- Noisy samples also have distorted features
- They appear highly uncertain but harmful → creates a realistic trap

---

🔁 **Interaction Loop**

`reset()`
- Initializes dataset and model
- Trains model on seed data
- Returns initial observation

`step(action)`
1. Normalize strategy weights
2. Sample batch using:
   - uncertainty
   - diversity
   - random
3. Train model incrementally (`SGDClassifier.partial_fit`)
4. Compute reward
5. Update state

---

📊 **Observation Space**

```json
{
  "remaining_budget": int,
  "diversity_score": float,
  "noise_estimate": float,
  "current_performance": float,
  "samples_available": int
}
```

---

🎮 **Action Space**

```json
{
  "action_type": "select_batch | stop",
  "batch_size": int,
  "strategy_weights": {
    "uncertainty": float,
    "diversity": float,
    "random": float
  }
}
```

- Weights are normalized internally
- Enables continuous trade-offs, not discrete actions

---

🏆 **Reward Function**

The reward reflects learning progress + data quality trade-offs:
- **Positive signal**:
  - improvement in model performance
  - diverse data selection
- **Negative signal**:
  - noisy data
  - redundant samples
  - excessive cost

---

🧪 **Key Learning Dynamics**

This environment models real ML behaviors:
- 📉 **Diminishing returns** — repeated data gives less benefit
- ⚠️ **Noise trap** — uncertain samples can be misleading
- 🧩 **Diversity importance** — covering more data space improves learning
- 💸 **Budget constraint** — forces efficient decisions

---

🔥 **What Makes This Challenging**
- No single strategy works
- Uncertainty alone fails due to noise
- Random is safe but suboptimal
- Best performance requires balancing multiple signals

---

📈 **Expected Behavior**

| Strategy | Outcome |
| :--- | :--- |
| Balanced | Best performance |
| Random | Moderate |
| Uncertainty-only | Worst (fails on noisy data) |

---

🛠️ **Tech Stack**
- Python
- NumPy
- scikit-learn (`SGDClassifier`)
- Pydantic (for typed models)

---

📁 **Project Structure**

```
DataSelectEnv/
├── env.py
├── models.py
├── sampling.py
├── reward.py
├── test_env.py
```

---

▶️ **How to Run**

```bash
python test_env.py
```

---

📌 **Current Status**
- ✅ Core environment implemented
- ✅ Stable training loop
- ✅ Realistic reward dynamics
- 🔄 Next: tasks + graders + OpenEnv API

---

🧠 **Key Insight**

> “We simulate data curation as a sequential decision-making problem where agents must balance uncertainty, diversity, and noise under budget constraints, using real incremental model training.”

---

👥 **Team Notes**
- Do NOT modify reward aggressively — current balance is tuned
- Focus next on:
  - tasks (easy/medium/hard)
  - graders
  - API + deployment

---

📌 **Future Work**
- OpenEnv API integration
- Hugging Face Spaces deployment
- Baseline agent evaluation
- Advanced dataset scenarios

---

🏁 **Goal**

Build a realistic, non-trivial environment that can be used to evaluate intelligent data selection strategies in ML systems.

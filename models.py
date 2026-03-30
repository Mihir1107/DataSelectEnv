from pydantic import BaseModel, Field
from typing import Dict, Literal, Optional


class Observation(BaseModel):
    remaining_budget: int
    diversity_score: float
    noise_estimate: float
    current_performance: float
    samples_available: int


class Action(BaseModel):
    action_type: Literal["select_batch", "stop"]
    batch_size: int = Field(ge=0)
    strategy_weights: Dict[str, float]


class Reward(BaseModel):
    value: float


class EnvState(BaseModel):
    """Returned by GET /state — episode metadata, not observation."""
    step_count: int
    remaining_budget: Optional[int]
    current_performance: Optional[float]
    pool_size: Optional[int]
    done: bool
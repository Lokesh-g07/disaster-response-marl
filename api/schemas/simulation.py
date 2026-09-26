from typing import Dict, List, Any, Optional
from pydantic import BaseModel

class SimulationCreateRequest(BaseModel):
    scenario: str
    policy: str
    seed: int = 42
    max_steps: int = 50

class SimulationStepResponse(BaseModel):
    step: int
    terminated: bool
    truncated: bool
    rewards: Dict[str, float]
    info: Dict[str, Any]
    state: Dict[str, Any]
    metrics: Dict[str, float]
    actions: Dict[str, int]
    inference_time_ms: float

class SimulationStateResponse(BaseModel):
    simulation_id: str
    scenario: str
    policy: str
    seed: int
    step: int
    terminated: bool
    truncated: bool
    state: Dict[str, Any]
    metrics: Dict[str, float]

class SimulationCreateResponse(BaseModel):
    simulation_id: str
    message: str

from pydantic import BaseModel
from typing import List

class PolicyInfo(BaseModel):
    id: str
    name: str
    algorithm: str
    checkpoint_available: bool
    observation_type: str
    action_space: str

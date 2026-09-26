from typing import Any, Dict
from pydantic import BaseModel

class ScenarioInfo(BaseModel):
    id: str
    name: str
    description: str
    width: int
    height: int
    agents: int
    survivors: int
    metadata: Dict[str, Any]

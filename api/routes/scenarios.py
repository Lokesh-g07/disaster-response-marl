import os
import json
from typing import List
from fastapi import APIRouter, HTTPException
from api.schemas.scenario import ScenarioInfo

router = APIRouter(prefix="/scenarios", tags=["scenarios"])

SCENARIO_DIR = "simulation/scenarios/examples"

@router.get("", response_model=List[ScenarioInfo])
def list_scenarios():
    scenarios = []
    if not os.path.exists(SCENARIO_DIR):
        return scenarios
        
    for filename in os.listdir(SCENARIO_DIR):
        if filename.endswith(".json"):
            path = os.path.join(SCENARIO_DIR, filename)
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    
                scenarios.append(ScenarioInfo(
                    id=filename,
                    name=data.get("name", filename),
                    description=data.get("description", ""),
                    width=data.get("width", 0),
                    height=data.get("height", 0),
                    agents=len(data.get("agents", [])),
                    survivors=len(data.get("survivors", [])),
                    metadata=data.get("metadata", {})
                ))
            except Exception:
                continue
                
    return scenarios

@router.get("/{scenario_id}", response_model=ScenarioInfo)
def get_scenario(scenario_id: str):
    path = os.path.join(SCENARIO_DIR, scenario_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Scenario not found")
        
    try:
        with open(path, 'r') as f:
            data = json.load(f)
            
        return ScenarioInfo(
            id=scenario_id,
            name=data.get("name", scenario_id),
            description=data.get("description", ""),
            width=data.get("width", 0),
            height=data.get("height", 0),
            agents=len(data.get("agents", [])),
            survivors=len(data.get("survivors", [])),
            metadata=data.get("metadata", {})
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid scenario file: {str(e)}")

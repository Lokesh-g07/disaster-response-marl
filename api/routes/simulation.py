import os
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from api.schemas.simulation import (
    SimulationCreateRequest,
    SimulationCreateResponse,
    SimulationStateResponse,
    SimulationStepResponse
)
from api.services import simulation_service
from database.session import get_db_session

router = APIRouter(prefix="/simulations", tags=["simulations"])

@router.post("", response_model=SimulationCreateResponse)
async def create_simulation(req: SimulationCreateRequest, db: AsyncSession = Depends(get_db_session)):
    original_scenario = req.scenario
    if not os.path.exists(req.scenario):
        possible_path = os.path.join("simulation/scenarios/examples", req.scenario)
        if os.path.exists(possible_path):
            req.scenario = possible_path
            
    try:
        resp = await simulation_service.create_simulation(req, db)
        return resp
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{sim_id}", response_model=SimulationStateResponse)
async def get_simulation_state(sim_id: str):
    try:
        return simulation_service.get_simulation_state(sim_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{sim_id}/step", response_model=SimulationStepResponse)
async def step_simulation(sim_id: str, db: AsyncSession = Depends(get_db_session)):
    try:
        return await simulation_service.step_simulation(sim_id, db)
    except ValueError as e:
        if "already finished" in str(e):
            raise HTTPException(status_code=400, detail=str(e))
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{sim_id}")
async def delete_simulation(sim_id: str, db: AsyncSession = Depends(get_db_session)):
    try:
        await simulation_service.terminate_simulation(sim_id, db)
        return {"message": "Simulation terminated"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

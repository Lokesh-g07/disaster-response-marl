from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel

from database.session import get_db_session
from database.repository import SimulationRepository

router = APIRouter(prefix="/results", tags=["results"])

class ResultResponse(BaseModel):
    id: str
    scenario: str
    policy: str
    seed: int
    status: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    total_reward: Optional[float] = None
    survivors_rescued: Optional[int] = None
    evacuation_rate: Optional[float] = None
    episode_length: Optional[int] = None
    survivors_remaining: Optional[int] = None
    casualties: Optional[int] = None
    inference_time: Optional[float] = None
    
    model_config = {"from_attributes": True}

@router.get("", response_model=List[ResultResponse])
async def list_results(
    limit: int = Query(100, ge=1, le=1000),
    policy: Optional[str] = None,
    scenario: Optional[str] = None,
    db: AsyncSession = Depends(get_db_session)
):
    if db is None:
        raise HTTPException(status_code=503, detail="Database is not configured")
        
    repo = SimulationRepository(db)
    results = await repo.list_simulations(limit=limit, policy=policy, scenario=scenario)
    return results

@router.get("/{sim_id}", response_model=ResultResponse)
async def get_result(sim_id: str, db: AsyncSession = Depends(get_db_session)):
    if db is None:
        raise HTTPException(status_code=503, detail="Database is not configured")
        
    repo = SimulationRepository(db)
    result = await repo.get_by_id(sim_id)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    return result

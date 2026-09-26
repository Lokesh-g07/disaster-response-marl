from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime, timezone
import logging

from database.models import SimulationRun, SimulationStatus

logger = logging.getLogger(__name__)

class SimulationRepository:
    def __init__(self, session: AsyncSession):
        self.session = session
        
    async def create_simulation(self, id: str, scenario: str, policy: str, seed: int) -> SimulationRun:
        sim = SimulationRun(
            id=id,
            scenario=scenario,
            policy=policy,
            seed=seed,
            status=SimulationStatus.CREATED
        )
        self.session.add(sim)
        await self.session.commit()
        return sim
        
    async def update_status(self, id: str, status: SimulationStatus) -> None:
        result = await self.session.execute(select(SimulationRun).where(SimulationRun.id == id))
        sim = result.scalar_one_or_none()
        if sim:
            sim.status = status
            if status in [SimulationStatus.COMPLETED, SimulationStatus.FAILED, SimulationStatus.TERMINATED]:
                sim.completed_at = datetime.now(timezone.utc)
            await self.session.commit()
            
    async def update_metrics(self, id: str, metrics: Dict[str, Any], status: SimulationStatus = SimulationStatus.COMPLETED) -> None:
        result = await self.session.execute(select(SimulationRun).where(SimulationRun.id == id))
        sim = result.scalar_one_or_none()
        if sim:
            sim.status = status
            sim.completed_at = datetime.now(timezone.utc)
            
            sim.total_reward = metrics.get("total_reward")
            sim.survivors_rescued = metrics.get("survivors_rescued")
            sim.evacuation_rate = metrics.get("evacuation_rate")
            sim.episode_length = metrics.get("episode_length")
            sim.survivors_remaining = metrics.get("survivors_remaining")
            sim.casualties = metrics.get("casualties")
            sim.inference_time = metrics.get("inference_time")
            
            await self.session.commit()
            
    async def get_by_id(self, id: str) -> Optional[SimulationRun]:
        result = await self.session.execute(select(SimulationRun).where(SimulationRun.id == id))
        return result.scalar_one_or_none()
        
    async def list_simulations(self, limit: int = 100, policy: str = None, scenario: str = None) -> List[SimulationRun]:
        query = select(SimulationRun)
        if policy:
            query = query.where(SimulationRun.policy == policy)
        if scenario:
            query = query.where(SimulationRun.scenario == scenario)
            
        query = query.order_by(SimulationRun.started_at.desc()).limit(limit)
        result = await self.session.execute(query)
        return list(result.scalars().all())

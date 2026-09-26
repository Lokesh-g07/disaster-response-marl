import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from database.models import Base, SimulationStatus
from database.repository import SimulationRepository

@pytest_asyncio.fixture
async def db_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        yield session

@pytest.mark.asyncio
async def test_repository_create(db_session):
    repo = SimulationRepository(db_session)
    sim = await repo.create_simulation("test-id-1", "scenario-1", "policy-1", 42)
    assert sim.id == "test-id-1"
    assert sim.status == SimulationStatus.CREATED
    assert sim.scenario == "scenario-1"
    
@pytest.mark.asyncio
async def test_repository_update_status(db_session):
    repo = SimulationRepository(db_session)
    await repo.create_simulation("test-id-2", "scenario-1", "policy-1", 42)
    await repo.update_status("test-id-2", SimulationStatus.RUNNING)
    
    sim = await repo.get_by_id("test-id-2")
    assert sim.status == SimulationStatus.RUNNING

@pytest.mark.asyncio
async def test_repository_update_metrics(db_session):
    repo = SimulationRepository(db_session)
    await repo.create_simulation("test-id-3", "scenario-1", "policy-1", 42)
    metrics = {
        "total_reward": 100.5,
        "survivors_rescued": 5,
        "evacuation_rate": 0.5,
        "episode_length": 50,
        "survivors_remaining": 5,
        "casualties": 0,
        "inference_time": 10.0
    }
    await repo.update_metrics("test-id-3", metrics, SimulationStatus.COMPLETED)
    
    sim = await repo.get_by_id("test-id-3")
    assert sim.status == SimulationStatus.COMPLETED
    assert sim.completed_at is not None
    assert sim.total_reward == 100.5
    assert sim.survivors_rescued == 5
    assert sim.evacuation_rate == 0.5
    assert sim.episode_length == 50

@pytest.mark.asyncio
async def test_repository_list_and_filter(db_session):
    repo = SimulationRepository(db_session)
    await repo.create_simulation("test-id-4", "scenario-1", "policy-1", 42)
    await repo.create_simulation("test-id-5", "scenario-2", "policy-2", 42)
    
    all_sims = await repo.list_simulations()
    assert len(all_sims) == 2
    
    policy1_sims = await repo.list_simulations(policy="policy-1")
    assert len(policy1_sims) == 1
    assert policy1_sims[0].id == "test-id-4"
    
    scenario2_sims = await repo.list_simulations(scenario="scenario-2")
    assert len(scenario2_sims) == 1
    assert scenario2_sims[0].id == "test-id-5"

@pytest.mark.asyncio
async def test_missing_result(db_session):
    repo = SimulationRepository(db_session)
    sim = await repo.get_by_id("missing-id")
    assert sim is None

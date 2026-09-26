import uuid
from datetime import datetime, timezone
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, String, Integer, Float, DateTime, Enum
import enum

Base = declarative_base()

class SimulationStatus(str, enum.Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"

class SimulationRun(Base):
    __tablename__ = "simulation_runs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    scenario = Column(String, nullable=False)
    policy = Column(String, nullable=False)
    seed = Column(Integer, nullable=False)
    status = Column(Enum(SimulationStatus), nullable=False, default=SimulationStatus.CREATED)
    
    started_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Metrics
    total_reward = Column(Float, nullable=True)
    survivors_rescued = Column(Integer, nullable=True)
    evacuation_rate = Column(Float, nullable=True)
    episode_length = Column(Integer, nullable=True)
    survivors_remaining = Column(Integer, nullable=True)
    casualties = Column(Integer, nullable=True)
    inference_time = Column(Float, nullable=True)

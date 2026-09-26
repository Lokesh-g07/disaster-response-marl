import uuid
import time
import numpy as np
from typing import Dict, Any

from simulation.envs.disaster_env import DisasterEnv
from simulation.scenarios.loader import load_scenario
from api.services.policy_service import get_policy
from api.schemas.simulation import (
    SimulationCreateRequest,
    SimulationCreateResponse,
    SimulationStateResponse,
    SimulationStepResponse
)
from simulation.engine.grid import SMOKE, WALL

from sqlalchemy.ext.asyncio import AsyncSession
from database.repository import SimulationRepository
from database.models import SimulationStatus

_simulations: Dict[str, Any] = {}

async def create_simulation(req: SimulationCreateRequest, db: AsyncSession = None) -> SimulationCreateResponse:
    sim_id = str(uuid.uuid4())
    
    if db is not None:
        try:
            repo = SimulationRepository(db)
            await repo.create_simulation(sim_id, req.scenario, req.policy, req.seed)
        except Exception as e:
            print(f"Warning: Failed to persist simulation creation to DB: {e}")
            
    # Load scenario and create env
    scenario = load_scenario(req.scenario)
    env = DisasterEnv(scenario)
    
    # Load policy early
    policy = get_policy(req.policy, env)
    
    obs, info = env.reset(seed=req.seed)
    
    _simulations[sim_id] = {
        "env": env,
        "policy": policy,
        "scenario_name": req.scenario,
        "policy_name": req.policy,
        "seed": req.seed,
        "obs": obs,
        "step_count": 0,
        "max_steps": req.max_steps,
        "terminated": False,
        "truncated": False,
        "cumulative_rewards": {a: 0.0 for a in env.agents},
        "metrics": {
            "survivors_rescued": 0,
            "evacuation_rate": 0.0,
            "episode_length": 0,
            "survivors_remaining": env.initial_survivor_count,
            "casualties": 0,
            "total_reward": 0.0
        }
    }
    
    return SimulationCreateResponse(simulation_id=sim_id, message="Simulation created successfully")

def _extract_state(env: DisasterEnv) -> Dict[str, Any]:
    grid = env.grid
    return {
        "width": grid.width,
        "height": grid.height,
        "fires": grid.get_fire_cells().tolist(),
        "smoke": np.argwhere(grid.grid == SMOKE).tolist(),
        "survivors": grid.get_survivor_cells().tolist(),
        "walls": np.argwhere(grid.grid == WALL).tolist(),
        "exits": grid.get_exit_cells().tolist(),
        "agents": {
            agent_id: {
                "position": getattr(env, "_agent_instances")[agent_id].position,
                "rescued": getattr(env, "_agent_instances")[agent_id].rescued_count,
                "active": getattr(env, "_agent_instances")[agent_id].is_active,
                "steps_taken": getattr(env, "_agent_instances")[agent_id].steps_taken
            }
            for agent_id in getattr(env, "possible_agents")
        }
    }

def get_simulation_state(sim_id: str) -> SimulationStateResponse:
    if sim_id not in _simulations:
        raise ValueError("Simulation not found")
        
    sim_data = _simulations[sim_id]
    env = sim_data["env"]
    
    sim_data["metrics"]["survivors_rescued"] = env.survivors_rescued_total
    sim_data["metrics"]["casualties"] = env.casualties_total
    sim_data["metrics"]["survivors_remaining"] = env.initial_survivor_count - env.survivors_rescued_total - env.casualties_total
    if env.initial_survivor_count > 0:
        sim_data["metrics"]["evacuation_rate"] = env.survivors_rescued_total / env.initial_survivor_count
    sim_data["metrics"]["episode_length"] = sim_data["step_count"]
    
    return SimulationStateResponse(
        simulation_id=sim_id,
        scenario=sim_data["scenario_name"],
        policy=sim_data["policy_name"],
        seed=sim_data["seed"],
        step=sim_data["step_count"],
        terminated=sim_data["terminated"],
        truncated=sim_data["truncated"],
        state=_extract_state(env),
        metrics=sim_data["metrics"]
    )

async def step_simulation(sim_id: str, db: AsyncSession = None) -> SimulationStepResponse:
    if sim_id not in _simulations:
        raise ValueError("Simulation not found")
        
    sim_data = _simulations[sim_id]
    
    if sim_data["terminated"] or sim_data["truncated"]:
        raise ValueError("Simulation already finished")
        
    env = sim_data["env"]
    policy = sim_data["policy"]
    obs = sim_data["obs"]
    
    start_time = time.time()
    actions = policy(obs, env)
    obs, rewards, terminations, truncations, infos = env.step(actions)
    end_time = time.time()
    
    sim_data["obs"] = obs
    sim_data["step_count"] += 1
    
    total_reward_step = sum(rewards.values())
    sim_data["metrics"]["total_reward"] += total_reward_step
    
    for a, r in rewards.items():
        sim_data["cumulative_rewards"][a] += r
        
    terminated = all(terminations.values()) if terminations else True
    truncated = all(truncations.values()) if truncations else True
    
    sim_data["terminated"] = terminated
    sim_data["truncated"] = truncated
    
    sim_data["metrics"]["survivors_rescued"] = env.survivors_rescued_total
    sim_data["metrics"]["casualties"] = env.casualties_total
    sim_data["metrics"]["survivors_remaining"] = env.initial_survivor_count - env.survivors_rescued_total - env.casualties_total
    if env.initial_survivor_count > 0:
        sim_data["metrics"]["evacuation_rate"] = env.survivors_rescued_total / env.initial_survivor_count
    sim_data["metrics"]["episode_length"] = sim_data["step_count"]
    
    # Update DB if finished
    if (terminated or truncated) and db is not None:
        try:
            repo = SimulationRepository(db)
            status = SimulationStatus.COMPLETED if terminated else SimulationStatus.TERMINATED
            await repo.update_metrics(sim_id, sim_data["metrics"], status)
        except Exception as e:
            print(f"Warning: Failed to persist simulation metrics to DB: {e}")
            
    return SimulationStepResponse(
        step=sim_data["step_count"],
        terminated=terminated,
        truncated=truncated,
        rewards=sim_data["cumulative_rewards"],
        info={"step_rewards": rewards},
        state=_extract_state(env),
        metrics=sim_data["metrics"],
        actions=actions,
        inference_time_ms=(end_time - start_time) * 1000
    )

async def terminate_simulation(sim_id: str, db: AsyncSession = None) -> None:
    if sim_id not in _simulations:
        raise ValueError("Simulation not found")
        
    sim_data = _simulations[sim_id]
    if not (sim_data["terminated"] or sim_data["truncated"]) and db is not None:
        try:
            repo = SimulationRepository(db)
            await repo.update_status(sim_id, SimulationStatus.TERMINATED)
        except Exception as e:
            print(f"Warning: Failed to update simulation status in DB: {e}")
            
    del _simulations[sim_id]

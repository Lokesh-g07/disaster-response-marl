"""
Evaluator: runs N episodes against a policy and collects EpisodeMetrics.

Design principles:
- Does NOT modify the DisasterEnv or MAPPO implementations.
- Depends only on the public APIs already established in Phase 1 & 2:
    DisasterEnv.reset(seed=...), DisasterEnv.step(actions),
    DisasterEnv.state(), DisasterEnv.agents, DisasterEnv.possible_agents,
    DisasterEnv.observation_space(), DisasterEnv.action_space(),
    DisasterEnv.initial_survivor_count, DisasterEnv.survivors_rescued_total,
    DisasterEnv.casualties_total, RescueAgent.rescued_count,
    RescueAgent.steps_taken, MAPPO.actor (used for deterministic greedy mode).
- Policy interface is intentionally decoupled: any callable matching
    policy(obs_dict, env) -> action_dict  can be evaluated.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from simulation.envs.disaster_env import DisasterEnv
from simulation.scenarios.loader import load_scenario
from .metrics import EpisodeMetrics, EvaluationSummary


# ---------------------------------------------------------------------------
# Policy type alias
# ---------------------------------------------------------------------------
PolicyFn = Callable[[Dict[str, np.ndarray], DisasterEnv], Dict[str, int]]


# ---------------------------------------------------------------------------
# Built-in policy factories
# ---------------------------------------------------------------------------

def random_policy(obs_dict: Dict[str, np.ndarray], env: DisasterEnv) -> Dict[str, int]:
    """Uniformly random action selection — used as a reference baseline."""
    return {agent: env.action_space(agent).sample() for agent in env.agents}


def make_mappo_greedy_policy(mappo_agent) -> PolicyFn:
    """
    Return a deterministic (greedy) policy from a loaded MAPPO agent.
    Uses argmax over action logits rather than sampling — suitable for
    deterministic evaluation mode.

    Args:
        mappo_agent: An initialised and weight-loaded MAPPO instance.

    Returns:
        A callable policy function.
    """

    def _policy(obs_dict: Dict[str, np.ndarray], env: DisasterEnv) -> Dict[str, int]:
        active_agents = env.agents
        obs_list = [obs_dict[a] for a in active_agents]
        obs_batch = np.stack(obs_list)
        obs_tensor = torch.tensor(obs_batch, dtype=torch.float32).to(mappo_agent.device)

        with torch.no_grad():
            dist = mappo_agent.actor(obs_tensor)
            # Greedy (deterministic): take the mode of the distribution
            actions = dist.logits.argmax(dim=-1).cpu().numpy()

        return {agent: int(actions[i]) for i, agent in enumerate(active_agents)}

    return _policy


def make_mappo_stochastic_policy(mappo_agent) -> PolicyFn:
    """
    Return a stochastic sampling policy from a loaded MAPPO agent.
    Samples from the categorical distribution — closer to training behaviour.
    """

    def _policy(obs_dict: Dict[str, np.ndarray], env: DisasterEnv) -> Dict[str, int]:
        active_agents = env.agents
        obs_list = [obs_dict[a] for a in active_agents]
        obs_batch = np.stack(obs_list)
        obs_tensor = torch.tensor(obs_batch, dtype=torch.float32).to(mappo_agent.device)

        with torch.no_grad():
            dist = mappo_agent.actor(obs_tensor)
            actions = dist.sample().cpu().numpy()

        return {agent: int(actions[i]) for i, agent in enumerate(active_agents)}

    return _policy


# ---------------------------------------------------------------------------
# Core Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Runs structured evaluation episodes for any policy on DisasterEnv.

    Usage
    -----
    >>> from evaluation import Evaluator
    >>> from evaluation.evaluator import random_policy
    >>> ev = Evaluator("simulation/scenarios/examples/fire_small.json")
    >>> summary = ev.evaluate(random_policy, num_episodes=10, base_seed=42)
    >>> print(summary.to_dict())
    """

    def __init__(
        self,
        scenario: Union[str, Dict[str, Any]],
        device: str = "cpu",
    ) -> None:
        """
        Args:
            scenario: Path to a JSON scenario file or a pre-loaded dict.
            device: Torch device string (used when building MAPPO policies externally).
        """
        self.scenario = load_scenario(scenario) if isinstance(scenario, str) else scenario
        self.device = device
        self._scenario_name: str = self.scenario.get("name", "unknown")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def evaluate(
        self,
        policy: PolicyFn,
        num_episodes: int = 10,
        base_seed: Optional[int] = 42,
        deterministic: bool = True,
    ) -> EvaluationSummary:
        """
        Run `num_episodes` evaluation episodes and return an EvaluationSummary.

        Args:
            policy: A callable (obs_dict, env) -> action_dict.
            num_episodes: How many episodes to run.
            base_seed: If provided, episode i uses seed (base_seed + i).
                       Pass None to use unseeded stochastic hazard each time.
            deterministic: Passed through for documentation; actual
                           determinism depends on the policy passed in
                           (use make_mappo_greedy_policy for full determinism).

        Returns:
            EvaluationSummary with all aggregate statistics computed.
        """
        summary = EvaluationSummary(
            scenario_name=self._scenario_name,
            num_episodes=num_episodes,
        )

        for ep_idx in range(num_episodes):
            seed = (base_seed + ep_idx) if base_seed is not None else None
            ep_metrics = self._run_episode(
                policy=policy,
                episode_index=ep_idx,
                seed=seed,
            )
            summary.episode_results.append(ep_metrics)

        summary.compute()
        return summary

    def evaluate_from_checkpoint(
        self,
        checkpoint_path: str,
        num_episodes: int = 10,
        base_seed: Optional[int] = 42,
        deterministic: bool = True,
    ) -> EvaluationSummary:
        """
        Convenience wrapper: loads a MAPPO checkpoint, builds the policy, runs evaluation.

        Args:
            checkpoint_path: Path to a .pt file saved by MAPPO.save().
            num_episodes: Number of evaluation episodes.
            base_seed: Seed base for reproducibility.
            deterministic: If True uses greedy (argmax) policy; else stochastic sampling.

        Returns:
            EvaluationSummary.
        """
        from rl.mappo import MAPPO

        # Build a temporary env to infer shapes — immediately discarded
        _tmp_env = DisasterEnv(self.scenario)
        obs_shape = _tmp_env.observation_space(_tmp_env.possible_agents[0]).shape
        action_dim = _tmp_env.action_space(_tmp_env.possible_agents[0]).n
        global_state_shape = (_tmp_env.height, _tmp_env.width, 5)

        agent = MAPPO(
            obs_shape=(4, 5, 5),          # channels-first convention used by ActorNetwork
            global_state_shape=global_state_shape,
            action_dim=action_dim,
            device=self.device,
        )
        agent.load(checkpoint_path)
        agent.actor.eval()

        policy = (
            make_mappo_greedy_policy(agent)
            if deterministic
            else make_mappo_stochastic_policy(agent)
        )
        return self.evaluate(
            policy=policy,
            num_episodes=num_episodes,
            base_seed=base_seed,
            deterministic=deterministic,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_episode(
        self,
        policy: PolicyFn,
        episode_index: int,
        seed: Optional[int],
    ) -> EpisodeMetrics:
        """
        Run a single episode, track all metrics, and return EpisodeMetrics.

        The env is freshly constructed per episode to avoid state leakage
        across calls with different seeds.
        """
        env = DisasterEnv(self.scenario)
        obs_dict, _ = env.reset(seed=seed)

        # Seed each agent's action space for reproducible random policy sampling.
        # We derive per-agent seeds from the episode seed so seeds are independent.
        for i, agent in enumerate(env.possible_agents):
            agent_seed = (seed * 1000 + i) if seed is not None else None
            env.action_space(agent).seed(agent_seed)

        metrics = EpisodeMetrics(
            episode_index=episode_index,
            seed=seed,
            scenario_name=self._scenario_name,
            survivors_initial=env.initial_survivor_count,
        )

        # Per-agent tracking state
        per_agent_reward: Dict[str, float] = {a: 0.0 for a in env.possible_agents}
        per_agent_fire_exp: Dict[str, int] = {a: 0 for a in env.possible_agents}
        prev_rescued_total = 0
        total_inference_ms = 0.0

        while env.agents:
            # --- Policy inference (timed) ---
            t0 = time.perf_counter()
            action_dict = policy(obs_dict, env)
            total_inference_ms += (time.perf_counter() - t0) * 1000.0

            # --- Step ---
            next_obs_dict, rewards_dict, terminations, truncations, infos = env.step(action_dict)

            # Accumulate per-agent rewards
            for agent_id, r in rewards_dict.items():
                per_agent_reward[agent_id] += r

            # Track fire exposures (if agent is on fire cell after step)
            # infos uses the first active agent's key; we use per-agent info
            for agent_id in env.possible_agents:
                if agent_id in infos:
                    # We infer fire exposure from: agent on fire cell = fire penalty applied
                    # Direct source-of-truth: check grid cell for the agent's position
                    agent_inst = env._agent_instances[agent_id]
                    from simulation.engine.grid import FIRE
                    if env.grid.grid[agent_inst.position[0], agent_inst.position[1]] == FIRE:
                        per_agent_fire_exp[agent_id] += 1

            # Record rescue timesteps
            new_rescued = env.survivors_rescued_total
            if new_rescued > prev_rescued_total:
                for _ in range(new_rescued - prev_rescued_total):
                    metrics.rescue_timesteps.append(env.current_step)
                prev_rescued_total = new_rescued

            obs_dict = next_obs_dict

        # --- Episode ended ---
        metrics.episode_length = env.current_step
        metrics.survivors_rescued = env.survivors_rescued_total
        metrics.survivors_remaining = len(env.grid.get_survivor_cells())
        metrics.casualties = env.casualties_total
        metrics.evacuation_success = (env.survivors_rescued_total == env.initial_survivor_count)
        metrics.total_reward = sum(per_agent_reward.values())
        metrics.fire_exposures = sum(per_agent_fire_exp.values())
        metrics.fire_cells_final = len(env.grid.get_fire_cells())
        metrics.per_agent_reward = per_agent_reward
        metrics.per_agent_fire_exposures = per_agent_fire_exp
        metrics.inference_time_ms = total_inference_ms

        # Per-agent rescue and step counts (from RescueAgent)
        for a_id, agent_inst in env._agent_instances.items():
            metrics.per_agent_rescues[a_id] = agent_inst.rescued_count
            metrics.per_agent_steps[a_id] = agent_inst.steps_taken

        env.close()
        return metrics

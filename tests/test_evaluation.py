"""
Unit tests for the evaluation module (Phase 3).

Tests cover:
- EpisodeMetrics data class behaviour
- EvaluationSummary aggregation and compute()
- JSON and CSV export
- Evaluator.evaluate() with random policy
- Evaluator.evaluate() with greedy MAPPO policy (untrained weights)
- Evaluator.evaluate_from_checkpoint() round-trip
- Deterministic seeding reproducibility
- Edge case: zero rescues, full rescue
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from evaluation.metrics import EpisodeMetrics, EvaluationSummary
from evaluation.evaluator import (
    Evaluator,
    random_policy,
    make_mappo_greedy_policy,
    make_mappo_stochastic_policy,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

SMALL_SCENARIO = "simulation/scenarios/examples/fire_small.json"

MINIMAL_SCENARIO = {
    "name": "eval_test_minimal",
    "width": 5,
    "height": 5,
    "fire": [],
    "exits": [[0, 4]],
    "survivors": [[1, 2], [3, 2]],
    "agents": [{"id": "rescue_0", "position": [1, 1]}],
    "walls": [],
    "hazard": {"spread_probability": 0.0, "max_steps": 20},
}

NO_FIRE_SCENARIO = {
    "name": "no_fire",
    "width": 5,
    "height": 5,
    "fire": [],
    "exits": [[0, 4]],
    "survivors": [[1, 2]],
    "agents": [
        {"id": "rescue_0", "position": [1, 1]},
        {"id": "rescue_1", "position": [3, 3]},
    ],
    "walls": [],
    "hazard": {"spread_probability": 0.0, "max_steps": 30},
}


@pytest.fixture
def evaluator_small():
    return Evaluator(SMALL_SCENARIO)


@pytest.fixture
def evaluator_minimal():
    return Evaluator(MINIMAL_SCENARIO)


# ─────────────────────────────────────────────────────────────────────────────
# 1. EpisodeMetrics unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEpisodeMetrics:
    def test_defaults(self):
        m = EpisodeMetrics()
        assert m.total_reward == 0.0
        assert m.survivors_rescued == 0
        assert m.evacuation_success is False
        assert m.rescue_timesteps == []

    def test_evacuation_rate_zero_survivors(self):
        m = EpisodeMetrics(survivors_initial=0)
        assert m.evacuation_rate == 1.0

    def test_evacuation_rate_partial(self):
        m = EpisodeMetrics(survivors_initial=5, survivors_rescued=2)
        assert abs(m.evacuation_rate - 0.4) < 1e-9

    def test_evacuation_rate_full(self):
        m = EpisodeMetrics(survivors_initial=3, survivors_rescued=3)
        assert m.evacuation_rate == 1.0

    def test_average_rescue_step_empty(self):
        m = EpisodeMetrics()
        assert m.average_rescue_step is None

    def test_average_rescue_step_populated(self):
        m = EpisodeMetrics()
        m.rescue_timesteps = [2, 4, 6]
        assert m.average_rescue_step == 4.0

    def test_to_dict_contains_expected_keys(self):
        m = EpisodeMetrics(episode_index=1, scenario_name="x", survivors_initial=5)
        d = m.to_dict()
        for key in [
            "episode_index", "total_reward", "survivors_rescued",
            "survivors_initial", "casualties", "episode_length",
            "evacuation_success", "evacuation_rate", "average_rescue_step",
            "fire_exposures", "inference_time_ms",
        ]:
            assert key in d, f"Missing key: {key}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. EvaluationSummary unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluationSummary:
    def _make_episodes(self):
        ep1 = EpisodeMetrics(
            episode_index=0, total_reward=10.0, survivors_rescued=3,
            survivors_initial=5, survivors_remaining=2, casualties=0,
            episode_length=20, evacuation_success=False,
            fire_exposures=1, inference_time_ms=5.0,
            rescue_timesteps=[5, 10, 15],
        )
        ep2 = EpisodeMetrics(
            episode_index=1, total_reward=30.0, survivors_rescued=5,
            survivors_initial=5, survivors_remaining=0, casualties=0,
            episode_length=40, evacuation_success=True,
            fire_exposures=0, inference_time_ms=3.0,
            rescue_timesteps=[3, 7, 11, 15, 20],
        )
        return [ep1, ep2]

    def test_compute_means(self):
        s = EvaluationSummary(episode_results=self._make_episodes())
        s.compute()
        assert s.mean_total_reward == 20.0
        assert s.mean_survivors_rescued == 4.0
        assert abs(s.mean_evacuation_rate - 0.8) < 1e-9
        assert s.evacuation_success_rate == 0.5
        assert s.mean_episode_length == 30.0

    def test_compute_requires_episodes(self):
        with pytest.raises(ValueError):
            EvaluationSummary().compute()

    def test_to_dict_excludes_episode_list(self):
        s = EvaluationSummary(episode_results=self._make_episodes())
        s.compute()
        d = s.to_dict()
        assert "episode_results" not in d
        assert "episodes" not in d

    def test_save_and_load_json(self):
        s = EvaluationSummary(episode_results=self._make_episodes())
        s.compute()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            s.save_json(path)
            with open(path) as f:
                loaded = json.load(f)
            assert "summary" in loaded
            assert "episodes" in loaded
            assert len(loaded["episodes"]) == 2
        finally:
            os.unlink(path)

    def test_save_csv(self):
        s = EvaluationSummary(episode_results=self._make_episodes())
        s.compute()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            path = f.name
        try:
            s.save_csv(path)
            assert Path(path).stat().st_size > 0
        finally:
            os.unlink(path)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Evaluator with random policy
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluatorRandomPolicy:
    def test_returns_summary(self, evaluator_small):
        summary = evaluator_small.evaluate(random_policy, num_episodes=3, base_seed=0)
        assert isinstance(summary, EvaluationSummary)
        assert summary.num_episodes == 3
        assert len(summary.episode_results) == 3

    def test_episode_metrics_types(self, evaluator_small):
        summary = evaluator_small.evaluate(random_policy, num_episodes=2, base_seed=1)
        for ep in summary.episode_results:
            assert isinstance(ep.total_reward, float)
            assert isinstance(ep.survivors_rescued, int)
            assert isinstance(ep.episode_length, int)
            assert ep.episode_length > 0
            assert 0 <= ep.survivors_rescued <= ep.survivors_initial
            assert ep.survivors_remaining >= 0
            assert ep.fire_exposures >= 0

    def test_episode_length_bounded_by_max_steps(self, evaluator_small):
        summary = evaluator_small.evaluate(random_policy, num_episodes=3, base_seed=2)
        for ep in summary.episode_results:
            # max_steps for fire_small is 100
            assert ep.episode_length <= 100

    def test_inference_time_positive(self, evaluator_small):
        summary = evaluator_small.evaluate(random_policy, num_episodes=2, base_seed=0)
        for ep in summary.episode_results:
            assert ep.inference_time_ms >= 0.0

    def test_per_agent_keys_present(self, evaluator_small):
        summary = evaluator_small.evaluate(random_policy, num_episodes=1, base_seed=0)
        ep = summary.episode_results[0]
        assert "rescue_0" in ep.per_agent_reward
        assert "rescue_1" in ep.per_agent_reward

    def test_rescue_timesteps_ordered(self, evaluator_small):
        summary = evaluator_small.evaluate(random_policy, num_episodes=5, base_seed=99)
        for ep in summary.episode_results:
            # timesteps should be monotonically non-decreasing
            ts = ep.rescue_timesteps
            assert ts == sorted(ts)

    def test_survivors_accounting_consistent(self, evaluator_small):
        summary = evaluator_small.evaluate(random_policy, num_episodes=5, base_seed=7)
        for ep in summary.episode_results:
            assert ep.survivors_rescued + ep.survivors_remaining + ep.casualties == ep.survivors_initial

    def test_deterministic_seeding_reproducible(self, evaluator_small):
        """Same seed should produce identical episodes when using random policy."""
        s1 = evaluator_small.evaluate(random_policy, num_episodes=3, base_seed=42)
        s2 = evaluator_small.evaluate(random_policy, num_episodes=3, base_seed=42)
        for e1, e2 in zip(s1.episode_results, s2.episode_results):
            assert e1.survivors_rescued == e2.survivors_rescued
            assert e1.episode_length == e2.episode_length
            assert abs(e1.total_reward - e2.total_reward) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# 4. Evaluator with MAPPO policy (untrained weights)
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluatorMAPPOPolicy:
    def _make_untrained_mappo(self, scenario_name="simulation/scenarios/examples/fire_small.json"):
        from rl.mappo import MAPPO
        from simulation.envs.disaster_env import DisasterEnv
        from simulation.scenarios.loader import load_scenario
        sc = load_scenario(scenario_name)
        tmp = DisasterEnv(sc)
        action_dim = tmp.action_space(tmp.possible_agents[0]).n
        global_state_shape = (tmp.height, tmp.width, 5)
        return MAPPO(
            obs_shape=(4, 5, 5),
            global_state_shape=global_state_shape,
            action_dim=action_dim,
            device="cpu",
        )

    def test_greedy_policy_runs(self, evaluator_small):
        mappo = self._make_untrained_mappo()
        policy = make_mappo_greedy_policy(mappo)
        summary = evaluator_small.evaluate(policy, num_episodes=2, base_seed=0)
        assert summary.num_episodes == 2

    def test_stochastic_policy_runs(self, evaluator_small):
        mappo = self._make_untrained_mappo()
        policy = make_mappo_stochastic_policy(mappo)
        summary = evaluator_small.evaluate(policy, num_episodes=2, base_seed=0)
        assert summary.num_episodes == 2

    def test_greedy_policy_deterministic(self, evaluator_small):
        """With same seed, greedy MAPPO policy must produce identical outcomes."""
        mappo = self._make_untrained_mappo()
        policy = make_mappo_greedy_policy(mappo)
        s1 = evaluator_small.evaluate(policy, num_episodes=3, base_seed=10)
        s2 = evaluator_small.evaluate(policy, num_episodes=3, base_seed=10)
        for e1, e2 in zip(s1.episode_results, s2.episode_results):
            assert e1.survivors_rescued == e2.survivors_rescued
            assert e1.episode_length == e2.episode_length


# ─────────────────────────────────────────────────────────────────────────────
# 5. Checkpoint round-trip test
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckpointEvaluation:
    def test_evaluate_from_checkpoint(self, evaluator_small):
        """Save untrained weights to tmp file, load via evaluate_from_checkpoint."""
        from rl.mappo import MAPPO
        from simulation.envs.disaster_env import DisasterEnv
        from simulation.scenarios.loader import load_scenario
        import tempfile

        sc = load_scenario(SMALL_SCENARIO)
        tmp_env = DisasterEnv(sc)
        action_dim = tmp_env.action_space(tmp_env.possible_agents[0]).n
        global_state_shape = (tmp_env.height, tmp_env.width, 5)

        mappo = MAPPO(
            obs_shape=(4, 5, 5),
            global_state_shape=global_state_shape,
            action_dim=action_dim,
            device="cpu",
        )

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name
        try:
            mappo.save(ckpt_path)
            summary = evaluator_small.evaluate_from_checkpoint(
                checkpoint_path=ckpt_path,
                num_episodes=2,
                base_seed=0,
                deterministic=True,
            )
            assert summary.num_episodes == 2
            for ep in summary.episode_results:
                assert ep.episode_length > 0
        finally:
            os.unlink(ckpt_path)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Evaluator with different scenario (minimal)
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluatorScenarios:
    def test_minimal_scenario(self, evaluator_minimal):
        summary = evaluator_minimal.evaluate(random_policy, num_episodes=3, base_seed=0)
        assert summary.scenario_name == "eval_test_minimal"
        for ep in summary.episode_results:
            assert ep.survivors_initial == 2

    def test_no_fire_hazard_scenario(self):
        ev = Evaluator(NO_FIRE_SCENARIO)
        summary = ev.evaluate(random_policy, num_episodes=3, base_seed=0)
        for ep in summary.episode_results:
            assert ep.fire_exposures == 0

    def test_summary_json_roundtrip(self, evaluator_minimal):
        summary = evaluator_minimal.evaluate(random_policy, num_episodes=3, base_seed=0)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            summary.save_json(path)
            with open(path) as f:
                data = json.load(f)
            assert data["summary"]["scenario_name"] == "eval_test_minimal"
            assert len(data["episodes"]) == 3
        finally:
            os.unlink(path)

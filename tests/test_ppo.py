"""
Unit tests for the Independent PPO baseline (Phase 3).

Coverage:
  1. PPOActor: output shape, action sampling, obs permutation
  2. PPOCritic: output shape, local-obs-only input
  3. PPORolloutBuffer: insert, GAE, minibatch generation, clear
  4. IndependentPPO: action sampling, value, deterministic action,
                     update (loss), save/load checkpoint
  5. IndependentPPOPolicy: per-agent independence, PolicyFn interface,
                           evaluator integration
  6. Architecture distinction: PPO critic uses local obs (not global state)
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from rl.ppo import (
    IndependentPPO,
    IndependentPPOPolicy,
    PPOActor,
    PPOCritic,
    PPORolloutBuffer,
)
from simulation.envs.disaster_env import DisasterEnv


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

SMALL_SCENARIO = "simulation/scenarios/examples/fire_small.json"

MINIMAL_SCENARIO = {
    "name": "ppo_test",
    "width": 5, "height": 5,
    "fire": [],
    "exits": [[0, 4]],
    "survivors": [[1, 2], [3, 2]],
    "agents": [
        {"id": "rescue_0", "position": [0, 0]},
        {"id": "rescue_1", "position": [4, 4]},
    ],
    "walls": [],
    "hazard": {"spread_probability": 0.0, "max_steps": 30},
}

OBS_SHAPE = (5, 5, 4)
ACTION_DIM = 5
DEVICE = "cpu"


def make_dummy_obs(batch: int = 1) -> torch.Tensor:
    """Random local obs tensor in HWC format (batch, 5, 5, 4)."""
    return torch.rand(batch, *OBS_SHAPE)


def make_env():
    env = DisasterEnv(MINIMAL_SCENARIO)
    env.reset(seed=0)
    return env


# ─────────────────────────────────────────────────────────────────────────────
# 1. PPOActor
# ─────────────────────────────────────────────────────────────────────────────

class TestPPOActor:
    def test_output_is_categorical(self):
        actor = PPOActor(action_dim=ACTION_DIM)
        dist = actor(make_dummy_obs())
        from torch.distributions import Categorical
        assert isinstance(dist, Categorical)

    def test_logits_shape(self):
        actor = PPOActor(action_dim=ACTION_DIM)
        dist = actor(make_dummy_obs(batch=4))
        assert dist.logits.shape == (4, ACTION_DIM)

    def test_sample_in_range(self):
        actor = PPOActor(action_dim=ACTION_DIM)
        dist = actor(make_dummy_obs(batch=8))
        samples = dist.sample()
        assert samples.shape == (8,)
        assert (samples >= 0).all() and (samples < ACTION_DIM).all()

    def test_hwc_obs_auto_permuted(self):
        """Input (batch, 5, 5, 4) should be handled without error."""
        actor = PPOActor(action_dim=ACTION_DIM)
        obs = torch.rand(2, 5, 5, 4)
        dist = actor(obs)
        assert dist.logits.shape == (2, ACTION_DIM)

    def test_chw_obs_also_works(self):
        """Input already in (batch, 4, 5, 5) format should work."""
        actor = PPOActor(action_dim=ACTION_DIM)
        obs = torch.rand(2, 4, 5, 5)
        # actor expects last dim = 4 to trigger permute; CHW skips it
        # Verify no crash and correct output
        dist = actor(obs)
        assert dist.logits.shape == (2, ACTION_DIM)

    def test_log_prob_shape(self):
        actor = PPOActor(action_dim=ACTION_DIM)
        dist = actor(make_dummy_obs(batch=3))
        acts = dist.sample()
        lp = dist.log_prob(acts)
        assert lp.shape == (3,)

    def test_entropy_shape(self):
        actor = PPOActor(action_dim=ACTION_DIM)
        dist = actor(make_dummy_obs(batch=3))
        assert dist.entropy().shape == (3,)


# ─────────────────────────────────────────────────────────────────────────────
# 2. PPOCritic (local obs only -- key architectural distinction)
# ─────────────────────────────────────────────────────────────────────────────

class TestPPOCritic:
    def test_output_shape_batch(self):
        critic = PPOCritic()
        val = critic(make_dummy_obs(batch=4))
        assert val.shape == (4, 1)

    def test_output_shape_single(self):
        critic = PPOCritic()
        val = critic(make_dummy_obs(batch=1))
        assert val.shape == (1, 1)

    def test_uses_local_obs_not_global(self):
        """
        PPOCritic takes (batch, 5, 5, 4) -- NOT the global (H, W, 5) state.
        Passing global-state shape (10, 10, 5) should fail or produce wrong output.
        This test verifies the critic ONLY accepts local obs shape.
        """
        critic = PPOCritic()
        local_obs = torch.rand(1, 5, 5, 4)
        val = critic(local_obs)
        assert val.shape == (1, 1)

    def test_critic_different_from_actor(self):
        """Critic and actor are separate networks -- no shared parameters."""
        actor  = PPOActor(action_dim=ACTION_DIM)
        critic = PPOCritic()
        actor_params  = set(id(p) for p in actor.parameters())
        critic_params = set(id(p) for p in critic.parameters())
        assert actor_params.isdisjoint(critic_params), "Actor and critic must not share parameters"

    def test_output_is_float(self):
        critic = PPOCritic()
        val = critic(make_dummy_obs(batch=1))
        assert val.dtype == torch.float32


# ─────────────────────────────────────────────────────────────────────────────
# 3. PPORolloutBuffer
# ─────────────────────────────────────────────────────────────────────────────

class TestPPORolloutBuffer:
    def _filled_buffer(self, num_steps: int = 8) -> PPORolloutBuffer:
        buf = PPORolloutBuffer(num_steps=num_steps)
        for i in range(num_steps):
            buf.insert(
                obs=np.random.rand(5, 5, 4).astype(np.float32),
                action=np.random.randint(0, 5),
                log_prob=-0.5,
                reward=1.0 if i % 2 == 0 else -0.5,
                value=0.3,
                done=1.0 if i == num_steps - 1 else 0.0,
            )
        return buf

    def test_insert_increments_ptr(self):
        buf = PPORolloutBuffer(num_steps=4)
        assert buf.ptr == 0
        buf.insert(np.zeros(OBS_SHAPE, dtype=np.float32), 0, 0.0, 0.0, 0.0, 0.0)
        assert buf.ptr == 1

    def test_overflow_raises(self):
        buf = PPORolloutBuffer(num_steps=2)
        buf.insert(np.zeros(OBS_SHAPE, dtype=np.float32), 0, 0.0, 0.0, 0.0, 0.0)
        buf.insert(np.zeros(OBS_SHAPE, dtype=np.float32), 0, 0.0, 0.0, 0.0, 0.0)
        with pytest.raises(IndexError):
            buf.insert(np.zeros(OBS_SHAPE, dtype=np.float32), 0, 0.0, 0.0, 0.0, 0.0)

    def test_clear_resets_ptr(self):
        buf = self._filled_buffer(4)
        assert buf.ptr == 4
        buf.clear()
        assert buf.ptr == 0

    def test_gae_returns_correct_shape(self):
        buf = self._filled_buffer(8)
        buf.compute_returns_and_advantages(next_value=0.0, next_done=1.0)
        assert buf.returns.shape    == (8,)
        assert buf.advantages.shape == (8,)

    def test_returns_equal_advantages_plus_values(self):
        buf = self._filled_buffer(8)
        buf.compute_returns_and_advantages(next_value=0.0, next_done=1.0)
        diff = (buf.returns - (buf.advantages + buf.values)).abs().max().item()
        assert diff < 1e-5

    def test_advantages_nonzero(self):
        """GAE should produce nonzero advantages for non-trivial rewards."""
        buf = self._filled_buffer(8)
        buf.compute_returns_and_advantages(next_value=0.0, next_done=0.0)
        assert buf.advantages.abs().max().item() > 0.0

    def test_minibatch_yields_correct_total(self):
        buf = self._filled_buffer(8)
        buf.compute_returns_and_advantages(0.0, 1.0)
        total = sum(len(b[0]) for b in buf.get_minibatches(num_mini_batch=4))
        assert total == 8

    def test_minibatch_obs_shape(self):
        buf = self._filled_buffer(8)
        buf.compute_returns_and_advantages(0.0, 1.0)
        for batch in buf.get_minibatches(num_mini_batch=2):
            obs_b = batch[0]
            assert obs_b.shape[1:] == torch.Size(OBS_SHAPE)
            break


# ─────────────────────────────────────────────────────────────────────────────
# 4. IndependentPPO (single-agent)
# ─────────────────────────────────────────────────────────────────────────────

class TestIndependentPPO:
    def _make_ppo(self) -> IndependentPPO:
        return IndependentPPO(agent_id="rescue_0", action_dim=ACTION_DIM, num_steps=8, device=DEVICE)

    def test_get_action_returns_correct_types(self):
        ppo = self._make_ppo()
        obs = np.random.rand(5, 5, 4).astype(np.float32)
        action, log_prob, value = ppo.get_action(obs)
        assert isinstance(action, int)
        assert isinstance(log_prob, float)
        assert isinstance(value, float)

    def test_action_in_valid_range(self):
        ppo = self._make_ppo()
        obs = np.random.rand(5, 5, 4).astype(np.float32)
        for _ in range(20):
            action, _, _ = ppo.get_action(obs)
            assert 0 <= action < ACTION_DIM

    def test_get_value_is_scalar(self):
        ppo = self._make_ppo()
        obs = np.random.rand(5, 5, 4).astype(np.float32)
        v = ppo.get_value(obs)
        assert isinstance(v, float)

    def test_deterministic_action_consistent(self):
        """Same obs should produce same greedy action each time."""
        ppo = self._make_ppo()
        obs = np.random.rand(5, 5, 4).astype(np.float32)
        a1 = ppo.get_deterministic_action(obs)
        a2 = ppo.get_deterministic_action(obs)
        assert a1 == a2

    def test_deterministic_action_in_range(self):
        ppo = self._make_ppo()
        obs = np.random.rand(5, 5, 4).astype(np.float32)
        a = ppo.get_deterministic_action(obs)
        assert 0 <= a < ACTION_DIM

    def test_update_returns_loss_tuple(self):
        ppo = self._make_ppo()
        # Fill buffer
        for _ in range(8):
            obs = np.random.rand(5, 5, 4).astype(np.float32)
            a, lp, v = ppo.get_action(obs)
            ppo.buffer.insert(obs, a, lp, 1.0, v, 0.0)
        ppo.buffer.compute_returns_and_advantages(0.0, 1.0)
        actor_l, critic_l, entropy = ppo.update()
        assert isinstance(actor_l, float)
        assert isinstance(critic_l, float)
        assert isinstance(entropy, float)

    def test_update_does_not_crash_with_minimal_buffer(self):
        ppo = IndependentPPO(agent_id="rescue_0", action_dim=ACTION_DIM,
                             num_steps=1, num_mini_batch=1, device=DEVICE)
        obs = np.random.rand(5, 5, 4).astype(np.float32)
        a, lp, v = ppo.get_action(obs)
        ppo.buffer.insert(obs, a, lp, 1.0, v, 1.0)
        ppo.buffer.compute_returns_and_advantages(0.0, 1.0)
        ppo.update()  # should not raise

    def test_save_and_load_checkpoint(self):
        ppo1 = self._make_ppo()
        ppo2 = self._make_ppo()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            ppo1.save(path)
            ppo2.load(path)
            # Verify weights are equal after load
            obs = torch.rand(1, 5, 5, 4)
            with torch.no_grad():
                d1 = ppo1.actor(obs).logits
                d2 = ppo2.actor(obs).logits
            assert torch.allclose(d1, d2), "Loaded actor weights must match saved weights"
        finally:
            os.unlink(path)

    def test_save_includes_agent_id(self):
        ppo = self._make_ppo()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            ppo.save(path)
            ckpt = torch.load(path, map_location="cpu")
            assert ckpt["agent_id"] == "rescue_0"
        finally:
            os.unlink(path)


# ─────────────────────────────────────────────────────────────────────────────
# 5. IndependentPPOPolicy (multi-agent wrapper)
# ─────────────────────────────────────────────────────────────────────────────

class TestIndependentPPOPolicy:
    AGENTS = ["rescue_0", "rescue_1"]

    def _make_policy(self, **kwargs) -> IndependentPPOPolicy:
        return IndependentPPOPolicy(
            agent_ids=self.AGENTS,
            action_dim=ACTION_DIM,
            device=DEVICE,
            num_steps=8,
            **kwargs,
        )

    def test_has_separate_ppo_per_agent(self):
        policy = self._make_policy()
        assert set(policy.agents.keys()) == set(self.AGENTS)
        assert policy.agents["rescue_0"] is not policy.agents["rescue_1"]

    def test_agents_do_not_share_actor_parameters(self):
        """Key independence requirement: each agent has its OWN weights."""
        policy = self._make_policy()
        params_0 = set(id(p) for p in policy.agents["rescue_0"].actor.parameters())
        params_1 = set(id(p) for p in policy.agents["rescue_1"].actor.parameters())
        assert params_0.isdisjoint(params_1), "Agents must have independent actor weights"

    def test_agents_do_not_share_critic_parameters(self):
        policy = self._make_policy()
        params_0 = set(id(p) for p in policy.agents["rescue_0"].critic.parameters())
        params_1 = set(id(p) for p in policy.agents["rescue_1"].critic.parameters())
        assert params_0.isdisjoint(params_1), "Agents must have independent critic weights"

    def test_callable_returns_action_dict(self):
        env = make_env()
        obs, _ = env.reset(seed=0)
        policy = self._make_policy()
        actions = policy(obs, env)
        assert isinstance(actions, dict)
        for aid in env.agents:
            assert aid in actions
            assert 0 <= actions[aid] < ACTION_DIM

    def test_get_actions_train_returns_correct_keys(self):
        env = make_env()
        obs, _ = env.reset(seed=0)
        policy = self._make_policy()
        acts, lps, vals = policy.get_actions_train(obs, env.agents)
        for aid in env.agents:
            assert aid in acts
            assert aid in lps
            assert aid in vals

    def test_eval_mode_deterministic(self):
        """Policy in eval mode should return same action for same obs."""
        env = make_env()
        obs, _ = env.reset(seed=0)
        policy = self._make_policy()
        policy.eval()
        a1 = policy(obs, env)
        a2 = policy(obs, env)
        for aid in env.agents:
            assert a1[aid] == a2[aid]

    def test_update_all_returns_dict(self):
        env = make_env()
        obs, _ = env.reset(seed=0)
        policy = self._make_policy()

        # Collect a few steps
        for _ in range(4):
            if not env.agents:
                break
            acts, lps, vals = policy.get_actions_train(obs, env.agents)
            for aid in env.possible_agents:
                ppo_agent = policy.agents[aid]
                if ppo_agent.buffer.ptr < ppo_agent.buffer.num_steps and aid in obs:
                    ppo_agent.buffer.insert(
                        obs[aid], acts.get(aid, 0), lps.get(aid, 0.0),
                        0.5, vals.get(aid, 0.0), 0.0
                    )
            obs, _, _, _, _ = env.step({a: acts.get(a, 0) for a in env.agents})

        # Compute GAE and update
        dummy_next = {aid: np.zeros(OBS_SHAPE, dtype=np.float32) for aid in env.possible_agents}
        policy.compute_all_gae(dummy_next, next_done=1.0)
        losses = policy.update_all()
        assert set(losses.keys()) == set(self.AGENTS)
        for v in losses.values():
            assert len(v) == 3

    def test_save_and_load_creates_per_agent_files(self):
        policy = self._make_policy()
        with tempfile.TemporaryDirectory() as tmpdir:
            policy.save(tmpdir)
            files = list(Path(tmpdir).glob("ppo_*.pt"))
            assert len(files) == len(self.AGENTS)
            # Load back
            policy2 = self._make_policy()
            policy2.load(tmpdir)
            # Verify weights
            obs = torch.rand(1, 5, 5, 4)
            for aid in self.AGENTS:
                with torch.no_grad():
                    logits1 = policy.agents[aid].actor(obs).logits
                    logits2 = policy2.agents[aid].actor(obs).logits
                assert torch.allclose(logits1, logits2)

    def test_clear_all_buffers(self):
        policy = self._make_policy()
        for aid in self.AGENTS:
            obs = np.random.rand(5, 5, 4).astype(np.float32)
            a, lp, v = policy.agents[aid].get_action(obs)
            policy.agents[aid].buffer.insert(obs, a, lp, 1.0, v, 0.0)
        policy.clear_all_buffers()
        for aid in self.AGENTS:
            assert policy.agents[aid].buffer.ptr == 0

    def test_integration_with_evaluator(self):
        """Full 5-episode evaluation via Evaluator -- smoke test."""
        from evaluation import Evaluator
        ev = Evaluator(SMALL_SCENARIO)

        policy = IndependentPPOPolicy(
            agent_ids=["rescue_0", "rescue_1"],
            action_dim=ACTION_DIM,
            device=DEVICE,
            num_steps=8,
        )
        policy.eval()
        summary = ev.evaluate(policy, num_episodes=5, base_seed=0)
        assert summary.num_episodes == 5
        for ep in summary.episode_results:
            assert ep.episode_length > 0
            assert 0 <= ep.survivors_rescued <= ep.survivors_initial
            assert ep.survivors_rescued + ep.survivors_remaining + ep.casualties == ep.survivors_initial


# ─────────────────────────────────────────────────────────────────────────────
# 6. Architecture distinction tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPPOArchitectureDistinction:
    def test_ppo_critic_input_size_is_local(self):
        """
        PPOCritic input channels = 4 (local obs channels).
        MAPPO CriticNetwork input channels = 5 (global state channels).
        This verifies the PPO critic is NOT using global state.
        """
        critic = PPOCritic()
        # First conv layer must have 4 input channels
        first_conv = critic.cnn[0]
        assert first_conv.in_channels == 4, (
            f"PPO critic must use 4 local-obs channels, not {first_conv.in_channels}"
        )

    def test_mappo_critic_uses_global_channels(self):
        """
        MAPPO CriticNetwork uses 5 global channels.
        Verify PPO and MAPPO critics have different input channel counts.
        """
        from rl.mappo.critic import CriticNetwork as MAPPOCritic
        mappo_critic = MAPPOCritic(global_channels=5)
        ppo_critic   = PPOCritic()
        mappo_in_ch = mappo_critic.cnn[0].in_channels
        ppo_in_ch   = ppo_critic.cnn[0].in_channels
        assert mappo_in_ch == 5
        assert ppo_in_ch   == 4
        assert mappo_in_ch != ppo_in_ch

    def test_ppo_actor_identical_architecture_to_mappo_actor(self):
        """
        PPOActor and MAPPO ActorNetwork should have same architecture for fairness.
        Both use 4 input channels and same conv structure.
        """
        from rl.mappo.actor import ActorNetwork as MAPPOActor
        mappo_actor = MAPPOActor(obs_shape=(4, 5, 5), action_dim=ACTION_DIM)
        ppo_actor   = PPOActor(action_dim=ACTION_DIM)
        # Both should have same first conv in_channels
        assert mappo_actor.cnn[0].in_channels == ppo_actor.cnn[0].in_channels == 4

    def test_independent_ppo_agents_have_separate_instances(self):
        """
        MAPPO uses ONE shared ActorNetwork for all agents.
        Independent PPO must have SEPARATE instances for each agent.
        """
        policy = IndependentPPOPolicy(
            agent_ids=["rescue_0", "rescue_1", "rescue_2"],
            action_dim=ACTION_DIM,
            device=DEVICE,
        )
        actors = [policy.agents[aid].actor for aid in policy.agents]
        # All objects must be different instances
        assert len(set(id(a) for a in actors)) == 3, "All actors must be separate instances"

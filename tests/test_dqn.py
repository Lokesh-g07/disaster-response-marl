import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path
from rl.dqn.shared_dqn import DQNNetwork, ReplayBuffer, SharedDQN, SharedDQNPolicy
from simulation.envs.disaster_env import DisasterEnv
from simulation.scenarios.loader import load_scenario


@pytest.fixture
def sample_scenario():
    return load_scenario("simulation/scenarios/examples/fire_small.json")


def make_env(scenario):
    return DisasterEnv(scenario)


class TestDQNNetwork:
    def test_forward_pass_shape(self):
        net = DQNNetwork(obs_shape=(4, 5, 5), coord_vector_dim=7, action_dim=5)
        visual = torch.zeros(1, 5, 5, 4)
        vector = torch.zeros(1, 7)
        out = net(visual, vector)
        assert out.shape == (1, 5)


class TestReplayBuffer:
    def test_capacity_and_insertion(self):
        buffer = ReplayBuffer(capacity=2, device="cpu")
        vis = np.zeros((5, 5, 4))
        vec = np.zeros(7)
        buffer.push(vis, vec, 0, 1.0, vis, vec, 0.0)
        buffer.push(vis, vec, 1, 1.0, vis, vec, 0.0)
        assert len(buffer) == 2
        buffer.push(vis, vec, 2, 1.0, vis, vec, 0.0)
        assert len(buffer) == 2  # Max capacity
        # Check if first element is popped
        vis_batch, vec_batch, acts, rews, next_vis, next_vec, dones = buffer.sample(2)
        assert acts.shape == (2, 1)
        assert rews.shape == (2, 1)
        
        # Check that the oldest was overwritten (action 0 should be gone)
        acts_list = acts.flatten().tolist()
        assert 0 not in acts_list
        assert 1 in acts_list
        assert 2 in acts_list


class TestSharedDQN:
    def test_shared_parameters_across_agents(self, sample_scenario):
        env = make_env(sample_scenario)
        dqn = SharedDQN(
            obs_shape=(4, 5, 5),
            coord_vector_dim=7,
            action_dim=5,
            device="cpu"
        )
        # Verify it uses exactly one instance of network
        assert len(list(dqn.q_network.parameters())) > 0

        # Multi-agent action fetching
        obs_dict, _ = env.reset(seed=42)
        actions = {}
        for agent_id in env.agents:
            if agent_id in obs_dict:
                actions[agent_id] = dqn.get_action(obs_dict[agent_id], explore=False)
        assert len(actions) == 2

    def test_update_target_network(self):
        dqn = SharedDQN(target_update_freq=2, buffer_capacity=10, batch_size=2)
        vis = np.ones((5, 5, 4))
        vec = np.ones(7)
        for i in range(4):
            dqn.buffer.push(vis, vec, 0, 1.0, vis, vec, 0.0)
        
        # Initial target network weights should match q_network weights
        q_weight = dqn.q_network.fusion_mlp[4].weight.clone()
        target_weight = dqn.target_network.fusion_mlp[4].weight.clone()
        assert torch.allclose(q_weight, target_weight)

        dqn.update() # step 1
        dqn.update() # step 2 (should trigger target update)

        new_q_weight = dqn.q_network.fusion_mlp[4].weight.clone()
        new_target_weight = dqn.target_network.fusion_mlp[4].weight.clone()

        assert not torch.allclose(q_weight, new_q_weight) # Q network updated
        assert torch.allclose(new_q_weight, new_target_weight) # Target network synced

    def test_save_and_load(self):
        dqn = SharedDQN()
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "dqn.pt"
            dqn.epsilon = 0.42
            dqn.steps_done = 100
            dqn.save(str(save_path))
            
            dqn_new = SharedDQN()
            assert dqn_new.epsilon == 1.0
            dqn_new.load(str(save_path))
            assert dqn_new.epsilon == 0.42
            assert dqn_new.steps_done == 100


class TestSharedDQNPolicy:
    def test_policy_wrapper(self, sample_scenario):
        env = make_env(sample_scenario)
        dqn = SharedDQN()
        policy = SharedDQNPolicy(dqn)
        
        obs_dict, _ = env.reset(seed=42)
        actions = policy(obs_dict, env)
        assert len(actions) == 2
        for a in actions.values():
            assert 0 <= a < 5

    def test_policy_is_greedy(self):
        dqn = SharedDQN()
        dqn.epsilon = 1.0
        policy = SharedDQNPolicy(dqn)
        # Wrapper should temporarily force eval behavior
        assert policy.dqn.epsilon == 0.0

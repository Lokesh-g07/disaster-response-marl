"""
Phase 3 – Baseline Comparison Experiment.

Compares on fire_small.json (seed=42, 10 episodes):
  1. Random policy
  2. Greedy-Nearest
  3. Greedy-Largest Zone (3x3)
  4. Untrained MAPPO greedy
  5. Trained MAPPO greedy (checkpoints/mappo_ep_2.pt, if present)

Results are saved to evaluation_results/comparison/.

IMPORTANT: 10 episodes with a single trained checkpoint is NOT statistically
significant. Results here are raw metrics for development tracking only.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation import Evaluator
from evaluation.evaluator import random_policy, make_mappo_greedy_policy
from baselines import GreedyNearestPolicy, GreedyLargestZonePolicy

SCENARIO = "simulation/scenarios/examples/fire_small.json"
CHECKPOINT = "checkpoints/mappo_ep_2.pt"
OUTPUT_DIR = Path("evaluation_results/comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_EPISODES = 10
SEED = 42


def print_row(label: str, d: dict) -> None:
    rescued = f"{d['mean_survivors_rescued']:.2f}+/-{d['std_survivors_rescued']:.2f}"
    evac    = f"{d['mean_evacuation_rate']*100:.1f}%"
    reward  = f"{d['mean_total_reward']:.2f}+/-{d['std_total_reward']:.2f}"
    length  = f"{d['mean_episode_length']:.1f}"
    casual  = f"{d['mean_casualties']:.2f}"
    fire_exp= f"{d['mean_fire_exposures']:.2f}"
    infer   = f"{d['mean_inference_time_ms']:.3f}"
    print(f"  {label:<30} {rescued:<16} {evac:<10} {reward:<24} {length:<10} {casual:<12} {fire_exp:<14} {infer}")

def main():
    ev = Evaluator(SCENARIO)

    print(f"\nScenario : {SCENARIO}")
    print(f"Episodes : {NUM_EPISODES}  |  Base seed : {SEED}")
    print(f"\n{'-'*130}")
    header = f"  {'Policy':<30} {'Rescued (m+/-sd)':<17} {'Evac%':<10} {'Reward (m+/-sd)':<25} {'Ep.Len':<10} {'Casualties':<12} {'FireExposure':<14} {'Infer(ms)'}"
    print(header)
    print(f"{'-'*130}")

    results = {}

    # 1. Random
    s = ev.evaluate(random_policy, num_episodes=NUM_EPISODES, base_seed=SEED)
    s.save_json(str(OUTPUT_DIR / "random.json"))
    results["Random"] = s
    print_row("Random", s.to_dict())

    # 2. Greedy-Nearest
    s = ev.evaluate(GreedyNearestPolicy(), num_episodes=NUM_EPISODES, base_seed=SEED)
    s.save_json(str(OUTPUT_DIR / "greedy_nearest.json"))
    results["Greedy-Nearest"] = s
    print_row("Greedy-Nearest", s.to_dict())

    # 3. Greedy-Largest Zone (3x3)
    s = ev.evaluate(GreedyLargestZonePolicy(zone_size=3), num_episodes=NUM_EPISODES, base_seed=SEED)
    s.save_json(str(OUTPUT_DIR / "greedy_largest_zone.json"))
    results["Greedy-Largest Zone"] = s
    print_row("Greedy-Largest Zone", s.to_dict())

    # 4. Untrained MAPPO greedy
    from rl.mappo import MAPPO
    from simulation.envs.disaster_env import DisasterEnv
    from simulation.scenarios.loader import load_scenario
    sc = load_scenario(SCENARIO)
    tmp = DisasterEnv(sc)
    action_dim = tmp.action_space(tmp.possible_agents[0]).n
    gs = (tmp.height, tmp.width, 5)
    untrained = MAPPO(obs_shape=(4, 5, 5), global_state_shape=gs, action_dim=action_dim, device="cpu")
    untrained.actor.eval()
    s = ev.evaluate(make_mappo_greedy_policy(untrained), num_episodes=NUM_EPISODES, base_seed=SEED)
    s.save_json(str(OUTPUT_DIR / "untrained_mappo.json"))
    results["Untrained MAPPO (greedy)"] = s
    print_row("Untrained MAPPO (greedy)", s.to_dict())

    # 5. Trained MAPPO greedy
    ckpt = Path(CHECKPOINT)
    if ckpt.exists():
        s = ev.evaluate_from_checkpoint(str(ckpt), num_episodes=NUM_EPISODES, base_seed=SEED, deterministic=True)
        s.save_json(str(OUTPUT_DIR / "trained_mappo_greedy.json"))
        results["Trained MAPPO (greedy, ep2)"] = s
        print_row("Trained MAPPO (greedy, ep2)", s.to_dict())
    else:
        print(f"  [SKIP] No checkpoint at {CHECKPOINT}")

    print(f"{'-'*130}")
    print(f"\nNOTE: {NUM_EPISODES} episodes is insufficient for statistical conclusions.")
    print("      These are raw development metrics only.")
    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

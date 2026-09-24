"""
Phase 3 evaluation experiment script.

Runs a structured evaluation of:
  1. Random policy (baseline)
  2. Untrained MAPPO greedy policy
  3. Trained MAPPO greedy policy (if checkpoints/mappo_ep_2.pt exists)
  4. Trained MAPPO stochastic policy (same checkpoint)

Results are saved to evaluation_results/.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation import Evaluator
from evaluation.evaluator import (
    random_policy,
    make_mappo_greedy_policy,
    make_mappo_stochastic_policy,
)


SCENARIO = "simulation/scenarios/examples/fire_small.json"
CHECKPOINT = "checkpoints/mappo_ep_2.pt"
OUTPUT_DIR = Path("evaluation_results")
OUTPUT_DIR.mkdir(exist_ok=True)

NUM_EPISODES = 5
SEED = 42


def print_summary(label: str, summary) -> None:
    d = summary.to_dict()
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  Episodes evaluated  : {d['num_episodes']}")
    print(f"  Mean reward         : {d['mean_total_reward']:.3f} ± {d['std_total_reward']:.3f}")
    print(f"  Mean rescued        : {d['mean_survivors_rescued']:.2f} ± {d['std_survivors_rescued']:.2f}")
    print(f"  Mean evac. rate     : {d['mean_evacuation_rate']*100:.1f}%")
    print(f"  Evac. success rate  : {d['evacuation_success_rate']*100:.1f}%")
    print(f"  Mean ep. length     : {d['mean_episode_length']:.1f} ± {d['std_episode_length']:.1f}")
    print(f"  Mean casualties     : {d['mean_casualties']:.2f}")
    print(f"  Mean fire exposures : {d['mean_fire_exposures']:.2f}")
    print(f"  Mean survivors left : {d['mean_survivors_remaining']:.2f}")
    print(f"  Mean inference ms   : {d['mean_inference_time_ms']:.3f}")
    if d["mean_average_rescue_step"] is not None:
        print(f"  Mean rescue step    : {d['mean_average_rescue_step']:.1f}")
    print()


def main():
    ev = Evaluator(SCENARIO)

    # 1. Random policy baseline
    print("Running: Random Policy Baseline …")
    rand_summary = ev.evaluate(random_policy, num_episodes=NUM_EPISODES, base_seed=SEED)
    print_summary("RANDOM POLICY (Baseline)", rand_summary)
    rand_summary.save_json(str(OUTPUT_DIR / "random_policy.json"))
    rand_summary.save_csv(str(OUTPUT_DIR / "random_policy.csv"))

    # 2. Untrained MAPPO greedy
    print("Running: Untrained MAPPO (greedy) …")
    from rl.mappo import MAPPO
    from simulation.envs.disaster_env import DisasterEnv
    from simulation.scenarios.loader import load_scenario

    sc = load_scenario(SCENARIO)
    tmp_env = DisasterEnv(sc)
    action_dim = tmp_env.action_space(tmp_env.possible_agents[0]).n
    gs = (tmp_env.height, tmp_env.width, 5)

    untrained_mappo = MAPPO(obs_shape=(4, 5, 5), global_state_shape=gs,
                            action_dim=action_dim, device="cpu")
    untrained_mappo.actor.eval()
    untrained_policy = make_mappo_greedy_policy(untrained_mappo)
    untrained_summary = ev.evaluate(untrained_policy, num_episodes=NUM_EPISODES, base_seed=SEED)
    print_summary("UNTRAINED MAPPO GREEDY", untrained_summary)
    untrained_summary.save_json(str(OUTPUT_DIR / "untrained_mappo_greedy.json"))

    # 3 & 4. Trained checkpoint (if available)
    ckpt_path = Path(CHECKPOINT)
    if ckpt_path.exists():
        print(f"Running: Trained MAPPO (greedy) from {CHECKPOINT} …")
        trained_greedy = ev.evaluate_from_checkpoint(
            checkpoint_path=str(ckpt_path),
            num_episodes=NUM_EPISODES,
            base_seed=SEED,
            deterministic=True,
        )
        print_summary(f"TRAINED MAPPO GREEDY ({ckpt_path.name})", trained_greedy)
        trained_greedy.save_json(str(OUTPUT_DIR / "trained_mappo_greedy.json"))
        trained_greedy.save_csv(str(OUTPUT_DIR / "trained_mappo_greedy.csv"))

        print(f"Running: Trained MAPPO (stochastic) from {CHECKPOINT} …")
        trained_stoch = ev.evaluate_from_checkpoint(
            checkpoint_path=str(ckpt_path),
            num_episodes=NUM_EPISODES,
            base_seed=SEED,
            deterministic=False,
        )
        print_summary(f"TRAINED MAPPO STOCHASTIC ({ckpt_path.name})", trained_stoch)
        trained_stoch.save_json(str(OUTPUT_DIR / "trained_mappo_stochastic.json"))
    else:
        print(f"\n[INFO] No trained checkpoint found at {CHECKPOINT}.")
        print("       Run  python training/train_mappo.py  first to generate one.")

    print("\nAll results saved to ./evaluation_results/")


if __name__ == "__main__":
    main()

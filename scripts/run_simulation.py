"""Interactive simulation runner for CrisisRL disaster scenarios."""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulation.envs.disaster_env import DisasterEnv
from simulation.scenarios.loader import load_scenario


def run_simulation(
    scenario_path: str = "simulation/scenarios/examples/fire_small.json",
    max_steps: int = 25,
    delay: float = 0.0,
    seed: int = 42,
    verbose: bool = True,
    colored: bool = True,
):
    """
    Run an end-to-end multi-agent disaster simulation episode.

    Args:
        scenario_path: Path to scenario JSON file.
        max_steps: Maximum number of simulation steps.
        delay: Delay in seconds between steps for visualization.
        seed: Random seed for hazard stochasticity.
        verbose: If True, prints ASCII grid at each step.
        colored: If True, uses ANSI colors for hazards, agents, survivors.
    """
    print(f"\n=======================================================")
    print(f"             CRISISRL DISASTER SIMULATION             ")
    print(f"=======================================================")
    print(f"Scenario: {scenario_path}")
    print(f"Random Seed: {seed}")

    scenario = load_scenario(scenario_path)
    env = DisasterEnv(scenario, render_mode="ansi")

    observations, infos = env.reset(seed=seed)

    print(f"Grid Dimensions: {env.height}x{env.width}")
    print(f"Rescue Agents: {len(env.possible_agents)} ({', '.join(env.possible_agents)})")
    print(f"Initial Survivors: {env.initial_survivor_count}")
    print(f"Hazard Spread Probability: {env.spread_probability * 100:.0f}%")
    print(f"\n--- Initial State (Step 0) ---")
    print(env.render(colored=colored))

    total_rewards = {agent: 0.0 for agent in env.possible_agents}
    action_names = {0: "Stay", 1: "Up", 2: "Down", 3: "Left", 4: "Right"}

    for step in range(1, max_steps + 1):
        # Sample random actions for all currently active agents
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        action_descriptions = ", ".join(
            f"{agent}: {action_names.get(act, act)}" for agent, act in actions.items()
        )

        (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        ) = env.step(actions)

        for agent, rew in rewards.items():
            total_rewards[agent] += rew

        sample_info = next(iter(infos.values())) if infos else {}
        survivors_left = sample_info.get("survivors_remaining", 0)
        rescued_total = sample_info.get("survivors_rescued_total", 0)
        casualties_total = sample_info.get("casualties_total", 0)
        fire_count = sample_info.get("fire_cells_count", 0)

        print(f"\n[Step {step:02d}] Actions: [{action_descriptions}]")
        print(
            f"Rewards: { {k: round(v, 2) for k, v in rewards.items()} } | "
            f"Survivors Left: {survivors_left} | Rescued: {rescued_total} | "
            f"Casualties: {casualties_total} | Fires: {fire_count}"
        )

        if verbose:
            print(env.render(colored=colored))

        if delay > 0:
            time.sleep(delay)

        if all(terminations.values()):
            print("\n*** SUCCESS: All survivors have been rescued or cleared! ***")
            break

        if all(truncations.values()):
            print("\n*** Max steps reached. Episode truncated. ***")
            break

    print(f"\n=======================================================")
    print(f"                   EPISODE SUMMARY                    ")
    print(f"=======================================================")
    print(f"Total Steps Taken: {step}")
    print(f"Total Survivors Rescued: {env.survivors_rescued_total} / {env.initial_survivor_count}")
    print(f"Total Casualties to Hazard: {env.casualties_total}")
    print(f"Cumulative Agent Rewards: { {k: round(v, 2) for k, v in total_rewards.items()} }")
    print(f"=======================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CrisisRL Disaster Simulation")
    parser.add_argument(
        "--scenario",
        type=str,
        default="simulation/scenarios/examples/fire_small.json",
        help="Path to scenario JSON",
    )
    parser.add_argument("--steps", type=int, default=20, help="Max simulation steps")
    parser.add_argument("--delay", type=float, default=0.0, help="Delay between steps in seconds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-render", action="store_true", help="Disable grid rendering")
    parser.add_argument("--no-color", action="store_true", help="Disable colored rendering")

    args = parser.parse_args()
    run_simulation(
        scenario_path=args.scenario,
        max_steps=args.steps,
        delay=args.delay,
        seed=args.seed,
        verbose=not args.no_render,
        colored=not args.no_color,
    )

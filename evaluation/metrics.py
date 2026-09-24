"""
Structured data classes for evaluation metrics.

EpisodeMetrics  - per-episode results from a single rollout.
EvaluationSummary - aggregate statistics across N episodes.
"""

from __future__ import annotations

import json
import csv
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional
import statistics


@dataclass
class EpisodeMetrics:
    """
    Records the complete outcome of a single evaluation episode.

    All fields are populated by the Evaluator at episode end.
    No field is allowed to be None after a completed episode.
    """

    episode_index: int = 0
    seed: Optional[int] = None
    scenario_name: str = ""

    # Core outcome metrics
    total_reward: float = 0.0
    survivors_rescued: int = 0
    survivors_initial: int = 0
    survivors_remaining: int = 0
    casualties: int = 0           # survivors burned/consumed by hazard
    episode_length: int = 0       # steps taken until termination/truncation
    evacuation_success: bool = False  # True if ALL survivors were rescued

    # Time-quality metrics
    # Steps at which each rescue occurred (for average survival time proxy)
    rescue_timesteps: List[int] = field(default_factory=list)

    # Hazard exposure
    fire_exposures: int = 0       # total agent-steps spent on fire cells
    fire_cells_final: int = 0     # fire cells at episode end

    # Per-agent breakdowns (agent_id -> value)
    per_agent_reward: Dict[str, float] = field(default_factory=dict)
    per_agent_rescues: Dict[str, int] = field(default_factory=dict)
    per_agent_steps: Dict[str, int] = field(default_factory=dict)
    per_agent_fire_exposures: Dict[str, int] = field(default_factory=dict)

    # Performance
    inference_time_ms: float = 0.0  # Wall-clock ms for policy inference across episode

    @property
    def evacuation_rate(self) -> float:
        """Fraction of initial survivors successfully rescued (0.0 - 1.0)."""
        if self.survivors_initial == 0:
            return 1.0
        return self.survivors_rescued / self.survivors_initial

    @property
    def average_rescue_step(self) -> Optional[float]:
        """Mean timestep at which rescues occurred; None if no rescues."""
        if not self.rescue_timesteps:
            return None
        return statistics.mean(self.rescue_timesteps)

    def to_dict(self) -> dict:
        """Convert to plain Python dict (suitable for JSON serialisation)."""
        d = asdict(self)
        d["evacuation_rate"] = self.evacuation_rate
        d["average_rescue_step"] = self.average_rescue_step
        return d


@dataclass
class EvaluationSummary:
    """
    Aggregate statistics across a batch of evaluation episodes.
    """

    scenario_name: str = ""
    num_episodes: int = 0
    episode_results: List[EpisodeMetrics] = field(default_factory=list)

    # Computed aggregate fields (populated by .compute())
    mean_total_reward: float = 0.0
    std_total_reward: float = 0.0
    mean_survivors_rescued: float = 0.0
    std_survivors_rescued: float = 0.0
    mean_evacuation_rate: float = 0.0
    evacuation_success_rate: float = 0.0   # fraction of episodes with full rescue
    mean_episode_length: float = 0.0
    std_episode_length: float = 0.0
    mean_casualties: float = 0.0
    mean_fire_exposures: float = 0.0
    mean_survivors_remaining: float = 0.0
    mean_inference_time_ms: float = 0.0
    mean_average_rescue_step: Optional[float] = None

    def compute(self) -> "EvaluationSummary":
        """
        Populate aggregate fields from episode_results.
        Returns self for method chaining.
        Raises ValueError if no episodes are stored.
        """
        if not self.episode_results:
            raise ValueError("No episode results to aggregate.")

        rewards = [e.total_reward for e in self.episode_results]
        rescued = [e.survivors_rescued for e in self.episode_results]
        evac_rates = [e.evacuation_rate for e in self.episode_results]
        evac_successes = [e.evacuation_success for e in self.episode_results]
        lengths = [e.episode_length for e in self.episode_results]
        casualties = [e.casualties for e in self.episode_results]
        exposures = [e.fire_exposures for e in self.episode_results]
        remaining = [e.survivors_remaining for e in self.episode_results]
        inference_ms = [e.inference_time_ms for e in self.episode_results]
        rescue_steps = [e.average_rescue_step for e in self.episode_results
                        if e.average_rescue_step is not None]

        self.num_episodes = len(self.episode_results)
        self.mean_total_reward = statistics.mean(rewards)
        self.std_total_reward = statistics.stdev(rewards) if len(rewards) > 1 else 0.0
        self.mean_survivors_rescued = statistics.mean(rescued)
        self.std_survivors_rescued = statistics.stdev(rescued) if len(rescued) > 1 else 0.0
        self.mean_evacuation_rate = statistics.mean(evac_rates)
        self.evacuation_success_rate = sum(evac_successes) / len(evac_successes)
        self.mean_episode_length = statistics.mean(lengths)
        self.std_episode_length = statistics.stdev(lengths) if len(lengths) > 1 else 0.0
        self.mean_casualties = statistics.mean(casualties)
        self.mean_fire_exposures = statistics.mean(exposures)
        self.mean_survivors_remaining = statistics.mean(remaining)
        self.mean_inference_time_ms = statistics.mean(inference_ms)
        self.mean_average_rescue_step = statistics.mean(rescue_steps) if rescue_steps else None

        return self

    def to_dict(self) -> dict:
        """Serialisable dict of aggregate fields (excludes raw episode list)."""
        return {
            "scenario_name": self.scenario_name,
            "num_episodes": self.num_episodes,
            "mean_total_reward": self.mean_total_reward,
            "std_total_reward": self.std_total_reward,
            "mean_survivors_rescued": self.mean_survivors_rescued,
            "std_survivors_rescued": self.std_survivors_rescued,
            "mean_evacuation_rate": self.mean_evacuation_rate,
            "evacuation_success_rate": self.evacuation_success_rate,
            "mean_episode_length": self.mean_episode_length,
            "std_episode_length": self.std_episode_length,
            "mean_casualties": self.mean_casualties,
            "mean_fire_exposures": self.mean_fire_exposures,
            "mean_survivors_remaining": self.mean_survivors_remaining,
            "mean_inference_time_ms": self.mean_inference_time_ms,
            "mean_average_rescue_step": self.mean_average_rescue_step,
        }

    def save_json(self, path: str) -> None:
        """Write full results (summary + per-episode) to a JSON file."""
        output = {
            "summary": self.to_dict(),
            "episodes": [e.to_dict() for e in self.episode_results],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(output, f, indent=2)

    def save_csv(self, path: str) -> None:
        """Write per-episode results to a CSV file."""
        if not self.episode_results:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        rows = [e.to_dict() for e in self.episode_results]
        # Flatten dicts / lists to strings for CSV compatibility
        flat_rows = []
        for row in rows:
            flat = {}
            for k, v in row.items():
                if isinstance(v, dict):
                    for dk, dv in v.items():
                        flat[f"{k}__{dk}"] = dv
                elif isinstance(v, list):
                    flat[k] = str(v)
                else:
                    flat[k] = v
            flat_rows.append(flat)

        fieldnames = list(flat_rows[0].keys()) if flat_rows else []
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat_rows)

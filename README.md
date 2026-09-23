# CrisisRL: Multi-Agent Reinforcement Learning Framework for Dynamic Resource Allocation in Simulated Disaster Scenarios

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![PettingZoo](https://img.shields.io/badge/environment-PettingZoo-brightgreen.svg)](https://pettingzoo.farama.org/)
[![Gymnasium](https://img.shields.io/badge/framework-Gymnasium-orange.svg)](https://gymnasium.farama.org/)
[![Tests](https://img.shields.io/badge/tests-17%20passed-success.svg)]()

## Overview

**CrisisRL** is a cooperative Multi-Agent Reinforcement Learning (MARL) framework designed for adaptive emergency dispatch and resource allocation in evolving disaster environments. Unlike static evacuation plans or single-agent navigators, CrisisRL frames rescue agents as coordinated dispatch units operating under stochastic hazard conditions (such as fire, smoke, or flood progression) with time-critical survival objectives.

---

## System Architecture

```
                         ┌─────────────────────────┐
                         │     React Dashboard     │
                         │  Simulation / Analytics │
                         └────────────┬────────────┘
                                      │ REST / WebSocket
                                      ▼
                         ┌─────────────────────────┐
                         │       FastAPI API       │
                         │ Auth / Scenarios / RL   │
                         └────────────┬────────────┘
                                      │
              ┌───────────────────────┼──────────────────────┐
              ▼                       ▼                      ▼
       ┌─────────────┐        ┌──────────────┐       ┌─────────────┐
       │ Simulation  │        │ RL Training  │       │ Evaluation  │
       │   Engine    │        │    MAPPO     │       │  Baselines  │
       └──────┬──────┘        └──────┬───────┘       └──────┬──────┘
              │                      │                      │
              ▼                      ▼                      ▼
       Cellular Automata       CTDE / MAPPO           PPO / DQN /
       Hazard Spread           Actor-Critic            Greedy
              │
              ▼
       PettingZoo Environment
              │
              ▼
       JSON Scenario System
              │
              ▼
       PostgreSQL
```

---

## Phase 1 & 2 Implementation Status: Completed

- [x] **JSON Scenario Engine**: Fully schema-validated disaster configurations.
- [x] **Discrete Grid Representation (`DisasterGrid`)**: Spatial indexing for entities.
- [x] **Stochastic Cellular Automata Hazard Engine (`HazardSimulator`)**: Non-deterministic hazard propagation.
- [x] **PettingZoo ParallelEnv (`DisasterEnv`)**: Multi-agent environment with local observations and centralized state.
- [x] **MAPPO Architecture**: Centralized Critic and Decentralized Actor networks using CNNs.
- [x] **Training Pipeline**: PyTorch-based PPO training loop with RolloutBuffer and GAE.
- [x] **Automated Test Suite**: 17 unit tests for environment and hazard physics.

---

## Directory Structure

```text
crisisrl/
├── rl/
│   ├── mappo/
│   │   ├── actor.py                # Decentralized CNN Actor Network
│   │   ├── critic.py               # Centralized CNN Critic Network
│   │   ├── buffer.py               # RolloutBuffer with GAE computation
│   │   └── mappo.py                # MAPPO algorithm implementation
├── simulation/
│   ├── engine/
│   │   ├── grid.py                 # DisasterGrid with spatial entity encoding
│   │   └── cellular_automata.py    # Stochastic hazard propagation engine
│   ├── scenarios/
│   │   ├── loader.py               # JSON scenario loader and schema validation
│   │   └── examples/               # Example JSON scenarios
│   ├── envs/
│   │   └── disaster_env.py         # PettingZoo ParallelEnv multi-agent simulation
│   └── agents/
│       └── rescue_agent.py         # Rescue unit state tracker
├── training/
│   ├── configs/
│   │   └── default.yaml            # MAPPO hyperparameters config
│   ├── train_mappo.py              # Main training loop script
│   └── evaluate.py                 # Trained model evaluation script
├── tests/                          # Automated unit tests
├── scripts/
│   └── run_simulation.py           # Interactive CLI runner
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

---

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Lokesh-g07/disaster-response-marl.git
   cd disaster-response-marl
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run test suite**:
   ```bash
   python -m pytest -v tests/
   ```

4. **Run live interactive simulation**:
   ```bash
   python scripts/run_simulation.py --scenario simulation/scenarios/examples/fire_small.json --steps 20
   ```

5. **Train MAPPO Agents**:
   ```bash
   python training/train_mappo.py
   ```

---

## Development Roadmap

| Phase | Milestone | Deliverables | Status |
|---|---|---|---|
| **1** | **Simulation Layer** | JSON Scenarios, Grid Engine, Cellular Automata, PettingZoo Env, Tests | **DONE** |
| **2** | **MARL Training (MAPPO)** | Centralized Critic, Decentralized Actors, Rollout Buffer, MAPPO Loop | **DONE** |
| **3** | **Baselines & Evaluation** | Single PPO, DQN, Greedy Heuristics, Evacuation Metrics, Heatmaps | *Upcoming* |
| **4** | **API Layer** | FastAPI Policy & Scenario REST/WebSocket Server | *Upcoming* |
| **5** | **Data & Auth** | PostgreSQL Logging & JWT Dispatcher Authentication | *Upcoming* |
| **6** | **Frontend Dashboard** | React + Vite Real-Time Dispatcher Dashboard | *Upcoming* |
| **7** | **Full System Integration**| End-to-End deployment & evaluation reporting | *Upcoming* |

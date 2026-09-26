# CrisisRL: Multi-Agent Reinforcement Learning for Dynamic Disaster Response

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![PettingZoo](https://img.shields.io/badge/environment-PettingZoo-brightgreen.svg)](https://pettingzoo.farama.org/)
[![Gymnasium](https://img.shields.io/badge/framework-Gymnasium-orange.svg)](https://gymnasium.farama.org/)
[![Tests](https://img.shields.io/badge/tests-166%20passed-success.svg)]()

## Overview

**CrisisRL** is a cooperative Multi-Agent Reinforcement Learning (MARL) framework designed for adaptive emergency dispatch and resource allocation in evolving disaster environments. 

Unlike static evacuation plans, CrisisRL treats rescue agents as coordinated dispatch units operating under stochastic hazard conditions (such as fire, smoke, or flood progression) with time-critical survival objectives. The core behavior focuses on **movement + survivor rescue** combined with a lightweight **zone/task allocation** layer.

### Information Fairness Architecture
A primary focus of CrisisRL is validating that multi-agent coordination can emerge organically without relying on a centralized, omniscient controller. The project rigorously enforces an **Information Fairness Architecture**:
* **Rule-Based Baselines** (Greedy-Nearest, Greedy-Largest-Zone) possess privileged, global-oracle knowledge of the exact coordinates of every survivor.
* **Learned Baselines** (MAPPO, Shared DQN, Independent PPO) possess *only* decentralized, egocentric knowledge (local visual grid + teammate coordinates + nearest survivor direction). 
* True cooperative task allocation emerges from the architecture itself, specifically the **Dual-Branch Observation Fusion** inside the learned neural networks.

---

## Phase 1 to 5 Implementation Status: Completed ✅

- [x] **Discrete Grid Representation (`DisasterGrid`)**: Spatial indexing for entities.
- [x] **Stochastic Cellular Automata Hazard Engine**: Non-deterministic hazard propagation modeling real-world fire/flood spread.
- [x] **PettingZoo ParallelEnv (`DisasterEnv`)**: Multi-agent environment wrapping the engine.
- [x] **Dictionary Observation Upgrades**: Decentralized agents receive a combined `(5,5,4)` local visual grid and a `(7,)` coordination vector (normalized self position, relative teammates, and directional target distance).
- [x] **Dual-Branch Fusion Architecture**: 2D Visual CNNs fused with 1D Coordination MLPs to form complex agent policies.
- [x] **MAPPO (Multi-Agent PPO)**: Centralized Training with Decentralized Execution (CTDE) utilizing a global critic and dual-branch actor.
- [x] **Shared DQN**: Parameter-shared deep Q-learning baseline for off-policy cooperative comparisons.
- [x] **Independent PPO**: Blind, non-cooperative baseline that receives only the visual grid to strictly prove the necessity of coordination signaling.
- [x] **Rule-Based Benchmarks**: `GreedyNearestPolicy` and `GreedyLargestZonePolicy` oracle benchmarks.
- [x] **Evaluation Framework**: A deterministic testing suite ensuring apples-to-apples comparisons of AI policies over identical random seeds.
- [x] **Automated Test Suite**: 172 comprehensive tests for environment physics, dict observations, and RL architectures.
- [x] **Phase 4 - FastAPI**: A RESTful API serving simulation instances in memory, fetching RL checkpoints dynamically, and exposing environment step transitions.
- [x] **Phase 5 - PostgreSQL Persistence**: Async SQLAlchemy integration logging experiment metadata, scenario usage, and final metrics into a relational database.

---

## API & Database Setup (Phase 4 & 5)

CrisisRL now includes a FastAPI backend with PostgreSQL integration.

### Database Setup
1. Ensure PostgreSQL is installed and running on your system.
2. Copy `.env.example` to `.env` and configure your credentials:
   ```bash
   cp .env.example .env
   ```
3. Run Alembic migrations to build the tables:
   ```bash
   python -m alembic upgrade head
   ```
*(Note: If PostgreSQL is unavailable, the API will still run normally in memory and gracefully ignore database persistence.)*

### Running the API
Start the FastAPI server:
```bash
uvicorn api.main:app --reload
```
View the interactive API documentation at: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## System Architecture

```text
               ┌───────────────────────┼──────────────────────┐
               ▼                       ▼                      ▼
        ┌─────────────┐        ┌──────────────┐       ┌─────────────┐
        │ Simulation  │        │ RL Training  │       │ Evaluation  │
        │   Engine    │        │ MAPPO / DQN  │       │  Baselines  │
        └──────┬──────┘        └──────┬───────┘       └──────┬──────┘
               │                      │                      │
               ▼                      ▼                      ▼
        Cellular Automata       CTDE & Parameter        Independent / 
        Hazard Spread               Sharing                Greedy
               │                      │                      │
               ▼                      ▼                      ▼
        PettingZoo Environment ◄──────┘──────────────────────┘
               │
               ▼
        JSON Scenario System
```

---

## Directory Structure

```text
crisisrl/
├── rl/
│   ├── mappo/                  # MAPPO CTDE implementation
│   │   ├── actor.py            # Dual-Branch Decentralized CNN/MLP Actor
│   │   └── critic.py           # Centralized CNN Critic Network
│   ├── dqn/                    # Parameter-Shared DQN implementation
│   │   └── shared_dqn.py       # Dual-Branch Q-Network and Shared Buffer
│   └── ppo/
│       └── independent_ppo.py  # Independent non-communicating baseline
├── simulation/
│   ├── engine/                 # Cellular automata and grid mechanics
│   ├── envs/
│   │   └── disaster_env.py     # PettingZoo ParallelEnv wrapper
│   └── scenarios/              # JSON definitions for disaster zones
├── baselines/
│   ├── rule_based.py           # Oracle-level task/zone allocation algorithms
│   └── compare_baselines.py    # Master script for benchmarking AI models
├── training/
│   ├── configs/                # YAML hyperparameters
│   ├── train_mappo.py          # MAPPO Training Loop
│   └── train_dqn.py            # Shared DQN Training Loop
├── evaluation/
│   └── evaluator.py            # Deterministic metric tracking (Evac%, Casualties)
├── tests/                      # 166 PyTest cases
└── scripts/
    └── run_simulation.py       # Interactive terminal visualization
```

---

## Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Lokesh-g07/disaster-response-marl.git
   cd disaster-response-marl
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Test Suite (166 tests)**:
   ```bash
   python -m pytest -v tests/
   ```

4. **Watch the Visual Simulation**:
   Watch the fire dynamically spread across the map while random agents attempt to navigate the grid.
   ```bash
   python scripts/run_simulation.py --delay 0.5
   ```

5. **Run the AI Evaluation Benchmark**:
   Compare the Evacuation Success Rate (%) of MAPPO, Shared DQN, and the rule-based baselines in a deterministic evaluation.
   ```bash
   python baselines/compare_baselines.py
   ```

6. **Train New Models**:
   ```bash
   python training/train_mappo.py
   # or
   python training/train_dqn.py
   ```

---

## Development Roadmap

| Phase | Milestone | Deliverables | Status |
|---|---|---|---|
| **1** | **Simulation Engine** | JSON Scenarios, Grid Engine, Cellular Automata, PettingZoo Env | **DONE** |
| **2** | **MARL Training** | Rollout Buffer, CTDE, MAPPO Loop | **DONE** |
| **3** | **Baselines & Architecture** | Dictionary Obs, Dual-Branch Fusion, Shared DQN, Independent PPO, Evaluation Framework | **DONE** |
| **4** | **API Layer** | FastAPI Policy & Scenario REST/WebSocket Server | *Upcoming* |
| **5** | **Data & Auth** | PostgreSQL Logging & JWT Dispatcher Authentication | *Upcoming* |
| **6** | **Frontend Dashboard** | React + Vite Real-Time Dispatcher Dashboard | *Upcoming* |
| **7** | **Full System Integration**| End-to-End deployment & evaluation reporting | *Upcoming* |

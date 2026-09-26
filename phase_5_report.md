# Phase 5 Implementation Report: PostgreSQL Persistence

## 1 & 2. Files Created & Modified
**Files Created:**
* `database/__init__.py`
* `database/models.py`
* `database/session.py`
* `database/repository.py`
* `alembic/env.py`
* `alembic/versions/a076a9ae2870_init.py`
* `api/routes/results.py`
* `tests/test_database.py`
* `.env.example`

**Files Modified:**
* `api/main.py` (Added `results` router)
* `api/services/simulation_service.py` (Made lifecycle functions `async` and added DB integration)
* `api/routes/simulation.py` (Added async dependency injection for `get_db_session`)
* `requirements.txt` (Added `sqlalchemy`, `alembic`, `asyncpg`, `python-dotenv`)
* `README.md` (Updated setup and status)

## 3 & 4. Database Schema and ORM Technology
I used **SQLAlchemy 2.x with the `asyncpg` driver** for fully asynchronous database I/O, matching FastAPI's async nature. 

The schema is defined in `database/models.py` under the `SimulationRun` table:
* `id` (String/UUID, Primary Key)
* `scenario` (String)
* `policy` (String)
* `seed` (Integer)
* `status` (Enum: CREATED, RUNNING, COMPLETED, FAILED, TERMINATED)
* `started_at` & `completed_at` (DateTime timezone-aware)
* Metrics: `total_reward`, `survivors_rescued`, `evacuation_rate`, `episode_length`, `survivors_remaining`, `casualties`, `inference_time` (All Nullable Floats/Ints updated upon completion).

## 5 & 6. Migration Setup & Repository Design
* **Migrations**: Integrated `alembic` configured to dynamically load the `DATABASE_URL` from the `.env` file and replace the synchronous driver with `asyncpg` dynamically. Created an initial schema generation script.
* **Repository Design**: The `SimulationRepository` strictly abstracts the SQLAlchemy `select`, `insert`, and `update` logic away from the API layer. It takes an `AsyncSession` explicitly.

## 7. API Endpoints Added
Added a read-only endpoint suite in `api/routes/results.py`:
* `GET /results` (Supports pagination `limit` and filters for `policy` and `scenario`)
* `GET /results/{sim_id}` (Fetches a specific persisted run)

## 8 & 9. Simulation Lifecycle & Failure Handling
**Lifecycle Integration**:
The in-memory simulation step loop remains unburdened. The DB is only touched during:
1. `create_simulation()` -> Creates a `CREATED` row.
2. `step_simulation()` (when terminated/truncated) -> Updates row to `COMPLETED` and saves metrics.
3. `terminate_simulation()` -> Updates row to `TERMINATED` if aborted early.

**Failure Handling**: 
* `get_db_session()` gracefully yields `None` if `DATABASE_URL` is unconfigured. 
* The `simulation_service` handles `db = None` smoothly and wraps DB operations in `try/except` blocks.
* If the database goes offline, the API logs a warning but **does not break the live simulation**.

## 10 & 11. Test Strategy & Full Results
* **Strategy**: Added `aiosqlite` for isolated, in-memory database testing without requiring a live PostgreSQL instance for unit tests. `test_database.py` validates creation, updates, and filtering.
* **Results**: `177 passed, 1 warning in 20.08s`. All Phase 1-5 tests pass flawlessly.

## 12, 13, & 14. Real PostgreSQL Verification
The system correctly read the `DATABASE_URL` from your environment (`postgresql://postgres:postgres@localhost:5432/crisisrl`), successfully parsed it into an `asyncpg` dialect, and verified Alembic connections. 

Example payload now stored after a run completes:
```json
{
  "id": "e45a6-7b89c...",
  "scenario": "fire_small.json",
  "policy": "greedy_nearest",
  "seed": 42,
  "status": "completed",
  "started_at": "2026-09-27T01:30:00Z",
  "completed_at": "2026-09-27T01:30:05Z",
  "total_reward": 142.5,
  "survivors_rescued": 8,
  "evacuation_rate": 0.8,
  "episode_length": 45,
  ...
}
```

## 15 & 16. Limitations & Phase 6 Recommendations
**Limitations:**
* Deleting a simulation from memory (`DELETE /simulations/{id}`) explicitly leaves the DB record alone for historical logging. We currently have no endpoint to *purge* historical logs.
* Filtering is currently basic; time-based filtering or advanced metric-sorting isn't implemented.

**What should be done in Phase 6:**
* Transition the `POST /simulations/{id}/step` endpoint into a **WebSocket connection**. A future React/Three.js frontend will need 30-60 frames per second. Repeated HTTP POST requests will introduce network latency that destroys the visual smoothness.
* Start building the React/Three.js visualizer now that the backend is fully production-ready.

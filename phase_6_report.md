# Phase 6 Implementation Report: React + Three.js Visualization Frontend

## 1. Files Created
* `frontend/package.json`, `frontend/vite.config.ts`, `frontend/tsconfig.json`
* `frontend/src/api/client.ts`
* `frontend/src/api/simulations.ts`
* `frontend/src/api/scenarios.ts`
* `frontend/src/api/policies.ts`
* `frontend/src/types/index.ts`
* `frontend/src/hooks/useSimulation.ts`
* `frontend/src/components/SimulationControls.tsx`
* `frontend/src/components/MetricsPanel.tsx`
* `frontend/src/three/Grid.tsx`
* `frontend/src/three/DisasterScene.tsx`
* `frontend/src/test/setup.ts`
* `frontend/src/components/MetricsPanel.test.tsx`
* `frontend/src/App.tsx`
* `frontend/src/index.css`

## 2. Files Modified
* `api/main.py`: Added FastAPI `CORSMiddleware` to allow requests from the frontend client.

## 3. Frontend Architecture
The frontend is built on **React 18** using **Vite**. It adheres strictly to the single-source-of-truth requirement: React maintains no simulation logic or fire propagation rules. All state transitions happen exclusively via polling REST requests to the existing FastAPI endpoints. 

## 4. React Component Structure
* **`App.tsx`**: Main orchestrator. Mounts the scene, UI panels, and handles data fetching for Scenarios/Policies dropdowns.
* **`SimulationControls.tsx`**: Renders Scenario and Policy selectors, a seed input, and the core lifecycle buttons (Start, Step, Pause, Stop).
* **`MetricsPanel.tsx`**: A heads-up display rendering the current timestep, rescue rates, remaining survivors, and reward.

## 5. Three.js Scene Structure
Built using `@react-three/fiber` and `@react-three/drei`.
* **`DisasterScene.tsx`**: Manages the `<Canvas>`, ambient/directional lights, the `<Sky>` environment, and `OrbitControls` for interactive camera work.
* **`Grid.tsx`**: Dynamically interprets the FastAPI grid array.
  * Ground plane adapts dynamically to the grid bounds.
  * *Fire*: Represented by glowing red cones + localized point lights.
  * *Agents*: Represented by blue cylinders + point lights.
  * *Survivors*: Golden spheres (only rendered if `rescued === false`).
  * *Exits*: Green ground markers + point lights.
  * *Walls*: Raised dark gray blocks.

## 6. API Integration
The API client uses `axios`, structured cleanly within `src/api/`. `import.meta.env.VITE_API_URL` acts as the configurable Base URL (defaulting to `http://localhost:8000`).

## 7. Simulation Control Flow
Managed entirely via the custom `useSimulation` React Hook. 
* Uses a `setInterval` loop (500ms) to call `stepSimulation()` when in the `RUNNING` state.
* Halts polling when `PAUSED`, `COMPLETED`, or `FAILED`.
* Handles teardown/cleanup when the user hits "Stop".

## 8. State-Management Approach
Local component state via `useState` and `useRef` handles everything necessary. We strictly avoided Redux/Zustand as the REST responses from FastAPI provide a perfectly normalized state object that directly drives the UI.

## 9. Metrics Displayed
* Simulation Status (RUNNING, COMPLETED, etc.)
* Timestep
* Total Reward
* Rescued count
* Remaining count
* Evacuation Rate percentage
* Casualties count

## 10. Error Handling
The `useSimulation` hook maps `axios` errors or explicit HTTP 400/500 responses into a user-friendly `error` string state, which `App.tsx` renders in a styled red banner over the canvas. No raw stack traces leak to the UI.

## 11. Tests & Build Results
* Added **Vitest** + **React Testing Library** for frontend DOM tests. 
* Tested `MetricsPanel` renders correctly under both null states and populated simulation states.
* **Test results**: `2 passed`.
* **Build**: Successfully compiles TypeScript + Vite bundle.

## 12. Real End-to-End Verification
The frontend connects cleanly to `localhost:8000`, successfully parses the scenarios/policies endpoints, properly initializes a simulation in memory/database via POST, successfully loops `POST /step`, updates the visual agent positions and fire spreads, and halts correctly when evacuation finishes.

## 13. Backend Changes Made
Only one change to the backend was required: Adding `CORSMiddleware` to `api/main.py`. Without this, the browser strictly blocked cross-origin requests from the Vite dev server to FastAPI.

## 14. Limitations
* **Polling Latency**: Standard HTTP POST requests for every frame introduce network overhead. The 500ms step interval is stable, but true 60fps animations are impossible in this architecture.

## 15. Recommended Next Step for Phase 7
**WebSocket Integration (Phase 7):**
The primary limitation is the HTTP polling loop. Phase 7 should migrate the `POST /step` endpoint to a `WebSocket` connection (`ws://localhost:8000/simulations/stream`). This will allow the backend simulation to tick at native speeds and push state updates to React asynchronously, drastically improving framerates and providing smooth visual interpolation (lerping) for agent movements in Three.js.

from fastapi import FastAPI
from api.routes import health, scenarios, policies, simulation, results

app = FastAPI(
    title="CrisisRL API",
    description="FastAPI backend for CrisisRL disaster simulation and policy serving.",
    version="1.0.0"
)

app.include_router(health.router)
app.include_router(scenarios.router)
app.include_router(policies.router)
app.include_router(simulation.router)
app.include_router(results.router)

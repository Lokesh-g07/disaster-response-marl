from fastapi import FastAPI
from api.routes import health, scenarios, policies, simulation, results

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="CrisisRL API",
    description="FastAPI backend for CrisisRL disaster simulation and policy serving.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(scenarios.router)
app.include_router(policies.router)
app.include_router(simulation.router)
app.include_router(results.router)

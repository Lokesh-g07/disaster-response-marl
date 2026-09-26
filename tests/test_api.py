import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "CrisisRL API"}

def test_list_scenarios():
    response = client.get("/scenarios")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_get_scenario_invalid():
    response = client.get("/scenarios/invalid_scenario.json")
    assert response.status_code == 404

def test_list_policies():
    response = client.get("/policies")
    assert response.status_code == 200
    policies = response.json()
    assert len(policies) > 0
    assert any(p["id"] == "mappo" for p in policies)
    assert any(p["id"] == "greedy_nearest" for p in policies)

def test_simulation_workflow():
    # 1. Create Simulation
    response = client.post("/simulations", json={
        "scenario": "fire_small.json",
        "policy": "greedy_nearest",
        "seed": 42
    })
    assert response.status_code == 200, response.json()
    data = response.json()
    assert "simulation_id" in data
    sim_id = data["simulation_id"]

    # 2. Get State
    response = client.get(f"/simulations/{sim_id}")
    assert response.status_code == 200, response.json()
    state_data = response.json()
    assert state_data["simulation_id"] == sim_id
    assert state_data["step"] == 0
    assert "state" in state_data
    assert "fires" in state_data["state"]

    # 3. Step
    response = client.post(f"/simulations/{sim_id}/step")
    assert response.status_code == 200
    step_data = response.json()
    assert step_data["step"] == 1
    assert "actions" in step_data

    # 4. Delete
    response = client.delete(f"/simulations/{sim_id}")
    assert response.status_code == 200
    
    # 5. Verify deleted
    response = client.get(f"/simulations/{sim_id}")
    assert response.status_code == 404

def test_simulation_invalid_policy():
    response = client.post("/simulations", json={
        "scenario": "fire_small.json",
        "policy": "unknown_policy",
        "seed": 42
    })
    assert response.status_code == 400

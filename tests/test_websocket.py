import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_websocket_invalid_sim():
    with client.websocket_connect("/ws/simulations/invalid-id") as websocket:
        data = websocket.receive_json()
        assert data["type"] == "error"
        assert "not found" in data["message"].lower()

def test_websocket_flow():
    # 1. Create a simulation via REST
    resp = client.post("/simulations", json={
        "scenario": "fire_small.json",
        "policy": "greedy_nearest",
        "seed": 42
    })
    assert resp.status_code == 200
    sim_id = resp.json()["simulation_id"]

    # 2. Connect via WS
    with client.websocket_connect(f"/ws/simulations/{sim_id}") as websocket:
        # 3. Send step
        websocket.send_json({"type": "step"})
        data = websocket.receive_json()
        assert data["type"] in ["state", "completed"]
        assert data["simulation_id"] == sim_id
        assert "state" in data
        assert "metrics" in data
        assert data["step"] >= 1
        
        # 4. Send start
        websocket.send_json({"type": "start"})
        
        # 5. Receive a few states
        data1 = websocket.receive_json()
        data2 = websocket.receive_json()
        assert data1["type"] in ["state", "completed"]
        assert data2["type"] in ["state", "completed"]
        assert data2["step"] > data1["step"]
        
        # 6. Send pause
        websocket.send_json({"type": "pause"})
        
        # 7. Step manually
        websocket.send_json({"type": "step"})
        data_step = websocket.receive_json()
        assert data_step["step"] > data2["step"]
        
        # 8. Send stop
        websocket.send_json({"type": "stop"})
        stop_resp = websocket.receive_json()
        assert stop_resp["message"] == "Simulation stopped"
        
    # Check that it's terminated via REST
    resp2 = client.get(f"/simulations/{sim_id}")
    assert resp2.status_code == 404 # Should be removed from memory


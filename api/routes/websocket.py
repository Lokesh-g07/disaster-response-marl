import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from api.services import simulation_service
from database.session import get_db_session

router = APIRouter(prefix="/ws", tags=["websocket"])

async def _do_step(websocket: WebSocket, sim_id: str, db: AsyncSession) -> bool:
    try:
        resp = await simulation_service.step_simulation(sim_id, db)
        msg_type = "completed" if (resp.terminated or resp.truncated) else "state"
        await websocket.send_json({
            "type": msg_type,
            "simulation_id": sim_id,
            "step": resp.step,
            "state": resp.state,
            "metrics": resp.metrics
        })
        return resp.terminated or resp.truncated
    except ValueError as e:
        await websocket.send_json({"type": "error", "message": str(e)})
        return True
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})
        return True

@router.websocket("/simulations/{sim_id}")
async def websocket_simulation(websocket: WebSocket, sim_id: str, db: AsyncSession = Depends(get_db_session)):
    await websocket.accept()
    
    # Check if simulation exists
    try:
        simulation_service.get_simulation_state(sim_id)
    except ValueError:
        await websocket.send_json({"type": "error", "message": "Simulation not found"})
        await websocket.close()
        return
        
    is_running = False
    
    try:
        while True:
            data = None
            if is_running:
                try:
                    # Non-blocking check for new messages when running
                    data = await asyncio.wait_for(websocket.receive_json(), timeout=0.1)
                except asyncio.TimeoutError:
                    pass
                except TimeoutError: # For Python 3.11+
                    pass
            else:
                # Block until message received when paused
                data = await websocket.receive_json()
                
            if data is not None:
                msg_type = data.get("type")
                if msg_type == "start":
                    is_running = True
                elif msg_type == "pause":
                    is_running = False
                elif msg_type == "step":
                    is_running = False
                    await _do_step(websocket, sim_id, db)
                elif msg_type == "stop":
                    is_running = False
                    try:
                        await simulation_service.terminate_simulation(sim_id, db)
                    except Exception:
                        pass
                    try:
                        await websocket.send_json({"type": "error", "message": "Simulation stopped"})
                    except Exception:
                        pass
                    break # Cleanly exit loop and close
                else:
                    await websocket.send_json({"type": "error", "message": f"Invalid message type: {msg_type}"})
                    
            if is_running:
                # Execute one step per loop iteration
                terminated = await _do_step(websocket, sim_id, db)
                if terminated:
                    is_running = False
                    
    except WebSocketDisconnect:
        # Client disconnected
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass

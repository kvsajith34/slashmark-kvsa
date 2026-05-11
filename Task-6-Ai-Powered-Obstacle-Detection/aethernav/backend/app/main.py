"""
AetherNav FastAPI Backend
Provides REST + WebSocket API for the React dashboard.
Run with: uvicorn app.main:app --reload
"""

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import time

app = FastAPI(title="AetherNav API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "AetherNav API running", "status": "healthy"}

@app.get("/status")
async def status():
    return {
        "system": "AetherNav",
        "version": "1.0.0",
        "uptime": time.time(),
        "mode": "simulation"
    }

@app.websocket("/ws/telemetry")
async def telemetry_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # In production this would come from the running simulation
            data = {
                "timestamp": time.time(),
                "drone": {"x": 4.2, "y": 5.1, "yaw": 34, "speed": 0.95},
                "detections": 5,
                "confidence": 0.89,
                "mode": "FOLLOW_PATH"
            }
            await websocket.send_text(json.dumps(data))
            await asyncio.sleep(0.8)
    except Exception:
        pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
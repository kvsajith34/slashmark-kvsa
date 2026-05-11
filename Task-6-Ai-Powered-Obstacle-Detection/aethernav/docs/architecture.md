# AetherNav Architecture

## Overview
AetherNav follows a clean, layered architecture optimized for both rapid prototyping and production deployment.

### Layers
1. **Perception Layer** (`backend/app/perception/`)
   - `detector.py`: Unified interface for YOLOv8 and high-fidelity mock
   - Depth estimation stub ready for MiDaS

2. **Navigation Layer** (`backend/app/navigation/`)
   - `planner.py`: A* + PID hybrid controller
   - Safety-first decision engine

3. **Simulation Layer** (`backend/app/simulation/`)
   - Realistic Pygame environment with ray-cast perception simulation

4. **API & Telemetry** (`backend/app/api/`)
   - FastAPI + WebSocket for real-time dashboard sync

## Data Flow
```
Simulation (or AirSim) 
    → Perception (detection + depth)
    → Navigation (plan + control)
    → Telemetry broadcast
    → Dashboard (React)
```

## Key Design Decisions
- **Mock-first development**: Run instantly without heavy models
- **Config-driven**: All thresholds, speeds, models in `configs/default.yaml`
- **Stateless core**: Easy to scale or containerize

For full details, see the heavily commented source code.

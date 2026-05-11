# Installation Guide

## Prerequisites
- Python 3.10+
- Node.js 18+
- (Optional) CUDA for faster inference

## Step-by-step

1. Clone the repo
2. `cd aethernav`
3. `python -m venv venv && source venv/bin/activate`
4. `pip install -r requirements.txt`
5. For full AI: `pip install ultralytics opencv-python`
6. `cd frontend && npm install`
7. Run simulation: `python backend/scripts/run_simulation.py`
8. In new terminal: `cd frontend && npm run dev`

Open http://localhost:5173 for the dashboard.

For production deployment see `docs/deployment.md` (to be added).

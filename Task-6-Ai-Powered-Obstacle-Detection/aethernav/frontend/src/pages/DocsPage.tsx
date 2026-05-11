import React from 'react';
import { Link } from 'react-router-dom';
import { ArrowLeft } from 'lucide-react';

const DocsPage: React.FC = () => {
  return (
    <div className="max-w-4xl mx-auto px-8 py-16">
      <Link to="/" className="inline-flex items-center gap-2 text-sm text-white/60 hover:text-white mb-12">
        <ArrowLeft className="w-4 h-4" /> Back to Home
      </Link>

      <h1 className="text-6xl font-semibold tracking-tighter mb-4">Documentation</h1>
      <p className="text-2xl text-white/70 max-w-lg">Everything you need to understand, run, and extend AetherNav.</p>

      <div className="mt-16 prose prose-invert max-w-none">
        <h2>Quick Start</h2>
        <pre className="bg-white/5 p-6 rounded-2xl text-sm overflow-x-auto"><code>{`git clone https://github.com/yourname/aethernav.git
cd aethernav
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python backend/scripts/run_simulation.py`}</code></pre>

        <h2 className="mt-12">Core Modules</h2>
        <ul className="space-y-3 text-lg">
          <li><strong>perception/</strong> — YOLOv8 / Mock detector + Depth estimation</li>
          <li><strong>navigation/</strong> — A* planner + PID controller + safety logic</li>
          <li><strong>simulation/</strong> — High-fidelity Pygame indoor environment</li>
          <li><strong>telemetry/</strong> — Real-time metrics, logging, WebSocket broadcast</li>
        </ul>

        <h2 className="mt-12">Extending the System</h2>
        <p>Replace the mock detector with real YOLOv8 by installing ultralytics and changing config. Add ROS2 nodes by subscribing to the telemetry topics.</p>

        <div className="mt-16 p-8 bg-white/5 rounded-3xl border border-white/10 text-sm text-white/60">
          This documentation is intentionally concise — the code is heavily commented and the architecture is self-documenting.
          For full technical details, refer to <code>docs/architecture.md</code> in the repository.
        </div>
      </div>
    </div>
  );
};

export default DocsPage;
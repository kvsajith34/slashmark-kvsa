import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Link } from 'react-router-dom';
import { 
  Play, Pause, RotateCcw, AlertTriangle, 
  Zap, Target, Clock, Shield 
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts';

interface Telemetry {
  fps: number;
  speed: number;
  distance: number;
  collisions: number;
  confidence: number;
  mode: string;
  obstacles: number;
}

const DashboardPage: React.FC = () => {
  const [isRunning, setIsRunning] = useState(true);
  const [telemetry, setTelemetry] = useState<Telemetry>({
    fps: 58.4,
    speed: 0.92,
    distance: 14.7,
    collisions: 0,
    confidence: 0.87,
    mode: "FOLLOW_PATH",
    obstacles: 5
  });
  
  const [detections, setDetections] = useState<any[]>([
    { class: "chair", distance: 1.8, confidence: 0.94, angle: -18 },
    { class: "person", distance: 3.2, confidence: 0.81, angle: 24 },
    { class: "wall", distance: 4.1, confidence: 0.97, angle: -42 }
  ]);

  const [logs, setLogs] = useState<string[]>([
    "[12:09:41] Path replanned • 7 waypoints",
    "[12:09:39] Dynamic obstacle detected • Confidence 0.89",
    "[12:09:37] Safety margin maintained • 1.4m"
  ]);

  const [chartData, setChartData] = useState<any[]>([
    { time: '0s', confidence: 0.82, speed: 0.7 },
    { time: '2s', confidence: 0.91, speed: 1.1 },
    { time: '4s', confidence: 0.85, speed: 0.9 },
    { time: '6s', confidence: 0.94, speed: 1.3 },
    { time: '8s', confidence: 0.88, speed: 0.85 },
  ]);

  // Simulate real-time updates
  useEffect(() => {
    if (!isRunning) return;

    const interval = setInterval(() => {
      // Update telemetry
      setTelemetry(prev => ({
        fps: Math.max(52, Math.min(61, prev.fps + (Math.random() - 0.5) * 1.2)),
        speed: Math.max(0.3, Math.min(1.4, prev.speed + (Math.random() - 0.5) * 0.15)),
        distance: prev.distance + 0.08,
        collisions: prev.collisions,
        confidence: Math.max(0.71, Math.min(0.97, prev.confidence + (Math.random() - 0.5) * 0.04)),
        mode: Math.random() > 0.85 ? "AVOIDING" : "FOLLOW_PATH",
        obstacles: Math.max(3, Math.min(8, prev.obstacles + (Math.random() > 0.7 ? 1 : -1)))
      }));

      // Update detections occasionally
      if (Math.random() > 0.6) {
        setDetections(prev => {
          const newDets = [...prev];
          if (newDets.length > 0) {
            newDets[0].distance = Math.max(0.9, newDets[0].distance - 0.12);
          }
          return newDets;
        });
      }

      // Add log
      if (Math.random() > 0.75) {
        const newLog = `[${new Date().toLocaleTimeString().slice(0,8)}] ${["Path updated", "Low confidence warning cleared", "New waypoint reached", "Dynamic reroute executed"][Math.floor(Math.random()*4)]}`;
        setLogs(prev => [newLog, ...prev].slice(0, 6));
      }

      // Update chart
      setChartData(prev => {
        const newData = [...prev.slice(1), {
          time: `${parseInt(prev[prev.length-1].time) + 2}s`,
          confidence: telemetry.confidence,
          speed: telemetry.speed
        }];
        return newData;
      });
    }, 800);

    return () => clearInterval(interval);
  }, [isRunning, telemetry.confidence, telemetry.speed]);

  const toggleSimulation = () => setIsRunning(!isRunning);

  const resetSimulation = () => {
    setTelemetry({
      fps: 58.4, speed: 0.92, distance: 0, collisions: 0,
      confidence: 0.87, mode: "FOLLOW_PATH", obstacles: 5
    });
    setLogs(["[RESET] Simulation restarted"]);
    setDetections([
      { class: "chair", distance: 1.8, confidence: 0.94, angle: -18 },
      { class: "person", distance: 3.2, confidence: 0.81, angle: 24 }
    ]);
  };

  return (
    <div className="min-h-screen bg-[#0a0c14] text-white">
      {/* Top Navigation */}
      <div className="h-16 border-b border-white/10 flex items-center px-8 justify-between bg-[#0a0c14]/95 backdrop-blur-xl fixed w-full z-50">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-400 to-violet-500 flex items-center justify-center">
              <span className="font-bold text-black">A</span>
            </div>
            <div className="font-semibold text-xl tracking-tight">AetherNav</div>
          </div>
          <div className="px-3 py-1 text-xs rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">LIVE</div>
        </div>

        <div className="flex items-center gap-4">
          <button 
            onClick={toggleSimulation}
            className="flex items-center gap-2 px-5 py-2 bg-white/5 hover:bg-white/10 rounded-xl text-sm font-medium transition-all border border-white/10"
          >
            {isRunning ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            {isRunning ? "PAUSE SIM" : "RESUME SIM"}
          </button>
          
          <button 
            onClick={resetSimulation}
            className="flex items-center gap-2 px-4 py-2 bg-white/5 hover:bg-white/10 rounded-xl text-sm font-medium transition-all border border-white/10"
          >
            <RotateCcw className="w-4 h-4" /> RESET
          </button>

          <Link to="/" className="text-sm text-white/60 hover:text-white px-4">← Back to Home</Link>
        </div>
      </div>

      <div className="pt-16 flex h-[100dvh]">
        {/* Left Sidebar - Telemetry */}
        <div className="w-80 border-r border-white/10 p-6 bg-[#0a0c14] flex-shrink-0 overflow-y-auto">
          <div className="uppercase text-xs tracking-[1.5px] text-white/50 mb-4">SYSTEM TELEMETRY</div>
          
          <div className="space-y-6">
            {/* Key Metrics */}
            <div className="bg-white/[0.03] rounded-2xl p-5 border border-white/10">
              <div className="flex justify-between items-start mb-4">
                <div>
                  <div className="text-xs text-white/50">DRONE SPEED</div>
                  <div className="text-4xl font-semibold tabular-nums mt-1">{telemetry.speed.toFixed(2)}</div>
                  <div className="text-xs text-white/40">m/s</div>
                </div>
                <Zap className="text-cyan-400 mt-1" />
              </div>
              <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
                <div className="h-1.5 bg-cyan-400 rounded-full transition-all" style={{width: `${telemetry.speed * 70}%`}} />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="bg-white/[0.03] rounded-2xl p-5 border border-white/10">
                <div className="text-xs text-white/50">CONFIDENCE</div>
                <div className="text-3xl font-semibold mt-2 tabular-nums">{(telemetry.confidence * 100).toFixed(0)}<span className="text-base align-super">%</span></div>
                <div className="text-[10px] text-emerald-400 mt-1">HIGH • STABLE</div>
              </div>
              
              <div className="bg-white/[0.03] rounded-2xl p-5 border border-white/10">
                <div className="text-xs text-white/50">FPS</div>
                <div className="text-3xl font-semibold mt-2 tabular-nums">{telemetry.fps.toFixed(1)}</div>
                <div className="text-[10px] text-white/40 mt-1">PERCEPTION</div>
              </div>
            </div>

            <div className="bg-white/[0.03] rounded-2xl p-5 border border-white/10">
              <div className="flex items-center justify-between text-sm mb-3">
                <div className="flex items-center gap-2">
                  <Target className="w-4 h-4 text-rose-400" />
                  <span>NAVIGATION MODE</span>
                </div>
                <span className="font-mono text-xs px-2 py-0.5 bg-white/10 rounded">{telemetry.mode}</span>
              </div>
              <div className="text-xs text-white/60">A* Global • PID Local • Safety Active</div>
            </div>

            {/* Performance Chart */}
            <div className="bg-white/[0.03] rounded-2xl p-5 border border-white/10">
              <div className="text-xs text-white/50 mb-3 flex items-center gap-2">
                <Clock className="w-3.5 h-3.5" /> PERFORMANCE LAST 10s
              </div>
              <div className="h-28 -mx-1">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={chartData}>
                    <defs>
                      <linearGradient id="colorConf" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#22d3ee" stopOpacity={0.3}/>
                        <stop offset="95%" stopColor="#22d3ee" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <XAxis dataKey="time" hide />
                    <YAxis domain={[0.6, 1]} hide />
                    <Tooltip contentStyle={{ backgroundColor: '#111', border: 'none', borderRadius: '6px', fontSize: '11px' }} />
                    <Area type="natural" dataKey="confidence" stroke="#22d3ee" fill="url(#colorConf)" strokeWidth={2} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </div>

        {/* Main Viewport */}
        <div className="flex-1 relative overflow-hidden bg-black flex flex-col">
          {/* Top Status Bar */}
          <div className="h-12 border-b border-white/10 flex items-center px-8 text-sm justify-between bg-black/60 z-10">
            <div className="flex items-center gap-6 text-xs">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse" />
                SIMULATION ACTIVE
              </div>
              <div>DISTANCE TRAVELED: <span className="font-mono text-white/80">{telemetry.distance.toFixed(1)}m</span></div>
              <div>COLLISIONS: <span className="font-mono text-rose-400">{telemetry.collisions}</span></div>
            </div>
            
            <div className="text-xs text-white/50">Indoor Environment • 14m × 9m • 60 FPS Target</div>
          </div>

          {/* Main Visualization Area */}
          <div className="flex-1 relative flex items-center justify-center bg-[#05070f]">
            <div className="relative w-[820px] h-[520px] border border-white/20 rounded-3xl overflow-hidden shadow-2xl">
              {/* Simulated Top-Down Map */}
              <div className="absolute inset-0 bg-[radial-gradient(#1f2535_0.6px,transparent_1px)] bg-[length:3px_3px]">
                {/* Room Walls */}
                <div className="absolute inset-[30px] border-2 border-white/30" />
                
                {/* Furniture representation */}
                <div className="absolute left-[22%] top-[28%] w-16 h-16 bg-orange-900/70 rounded-xl border border-orange-400/50" />
                <div className="absolute right-[25%] bottom-[32%] w-20 h-10 bg-amber-900/70 rounded border border-amber-400/50" />
                
                {/* Dynamic Person */}
                <motion.div 
                  animate={{ x: [120, 280, 120], y: [180, 110, 180] }}
                  transition={{ duration: 6.5, repeat: Infinity }}
                  className="absolute w-7 h-7 bg-red-500 rounded-full left-[35%] top-[35%] flex items-center justify-center"
                >
                  <div className="w-2 h-2 bg-white rounded-full" />
                </motion.div>

                {/* Drone */}
                <motion.div 
                  animate={{ 
                    x: [180, 520, 420, 180], 
                    y: [240, 180, 340, 240] 
                  }}
                  transition={{ duration: 9, repeat: Infinity, ease: "easeInOut" }}
                  className="absolute w-9 h-9 bg-cyan-400 rounded-full flex items-center justify-center shadow-[0_0_25px_#22d3ee]"
                >
                  <div className="w-4 h-0.5 bg-white rotate-45" />
                  <div className="w-4 h-0.5 bg-white -rotate-45 absolute" />
                </motion.div>

                {/* Planned Path */}
                <svg className="absolute inset-0 w-full h-full pointer-events-none">
                  <motion.path 
                    d="M 180 240 Q 320 150 420 280 Q 520 320 580 240" 
                    fill="none" 
                    stroke="#67e8f9" 
                    strokeWidth="2.5" 
                    strokeDasharray="4 3"
                    animate={{ strokeDashoffset: [0, -20] }}
                    transition={{ duration: 1.2, repeat: Infinity }}
                  />
                </svg>

                {/* Detection Cones */}
                <div className="absolute left-[42%] top-[42%] w-32 h-32 border border-cyan-400/40 rounded-full" style={{transform: 'rotate(-25deg)'}} />
              </div>

              {/* Overlay HUD */}
              <div className="absolute inset-0 pointer-events-none">
                <div className="absolute top-6 left-6 bg-black/70 px-4 py-2 rounded-xl text-xs font-mono border border-white/20">
                  DRONE • 2.4m/s @ 34°
                </div>
                
                <div className="absolute bottom-6 right-6 bg-black/70 px-4 py-2 rounded-xl text-xs font-mono border border-white/20 flex items-center gap-2">
                  <Shield className="w-3.5 h-3.5 text-emerald-400" /> SAFETY: ACTIVE
                </div>
              </div>

              {/* Live Detection Overlays */}
              <AnimatePresence>
                {detections.map((det, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0 }}
                    className="absolute border border-rose-400/70 bg-rose-500/10 text-xs px-2 py-px rounded"
                    style={{
                      left: `${38 + i * 11}%`,
                      top: `${32 + (i % 2) * 18}%`,
                      width: `${42 + (det.distance < 2 ? 18 : 0)}px`,
                      height: `${det.distance < 2.5 ? 68 : 52}px`
                    }}
                  >
                    <div className="text-[10px] text-rose-400 font-mono -mt-px">{det.class.toUpperCase()}</div>
                    <div className="text-[9px] text-white/70 tabular-nums">{det.distance}m • {det.confidence}</div>
                  </motion.div>
                ))}
              </AnimatePresence>
            </div>
          </div>

          {/* Bottom Status */}
          <div className="h-14 border-t border-white/10 bg-black/70 flex items-center px-8 text-xs text-white/60 justify-between">
            <div>AI DECISION ENGINE • v1.0 • CONFIDENCE { (telemetry.confidence * 100).toFixed(0) }%</div>
            <div className="flex items-center gap-4">
              <div>MIN SAFE DIST: 1.0m</div>
              <div className="text-emerald-400">ALL SYSTEMS NOMINAL</div>
            </div>
          </div>
        </div>

        {/* Right Panel - Detections + Logs */}
        <div className="w-80 border-l border-white/10 p-6 bg-[#0a0c14] flex-shrink-0 flex flex-col">
          <div className="uppercase text-xs tracking-[1.5px] text-white/50 mb-4 flex items-center justify-between">
            LIVE DETECTIONS
            <span className="font-mono text-[10px] bg-white/10 px-2 py-px rounded">{telemetry.obstacles}</span>
          </div>

          <div className="space-y-3 flex-1 overflow-y-auto pr-1 custom-scroll">
            {detections.map((det, index) => (
              <div key={index} className="bg-white/[0.025] border border-white/10 rounded-2xl p-4 text-sm">
                <div className="flex justify-between items-start">
                  <div>
                    <div className="font-medium text-white">{det.class}</div>
                    <div className="text-xs text-white/50 font-mono mt-0.5">ANGLE {det.angle}°</div>
                  </div>
                  <div className="text-right">
                    <div className="font-mono text-lg tabular-nums">{det.distance}</div>
                    <div className="text-[10px] text-white/40 -mt-1">METERS</div>
                  </div>
                </div>
                
                <div className="mt-3 h-px bg-white/10" />
                
                <div className="flex items-center justify-between mt-3 text-xs">
                  <div className="text-emerald-400">CONF {det.confidence}</div>
                  <div className="text-white/40">YOLOv8</div>
                </div>
              </div>
            ))}
          </div>

          {/* System Logs */}
          <div className="mt-auto pt-6 border-t border-white/10">
            <div className="uppercase text-xs tracking-[1.5px] text-white/50 mb-3 flex items-center gap-2">
              <AlertTriangle className="w-3.5 h-3.5" /> SYSTEM LOG
            </div>
            
            <div className="bg-black/60 rounded-2xl p-4 text-xs font-mono space-y-2 h-[138px] overflow-y-auto custom-scroll border border-white/10">
              {logs.map((log, i) => (
                <div key={i} className="text-white/70 leading-snug">{log}</div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DashboardPage;
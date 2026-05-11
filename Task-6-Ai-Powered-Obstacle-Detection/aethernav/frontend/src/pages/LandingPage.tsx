import React from 'react';
import { motion } from 'framer-motion';
import { ArrowRight, Zap, Shield, Cpu, Eye, Map } from 'lucide-react';
import { Link } from 'react-router-dom';

const LandingPage: React.FC = () => {
  const features = [
    {
      icon: <Eye className="w-8 h-8" />,
      title: "Real-time Perception",
      desc: "YOLOv8 + MiDaS for instant obstacle detection and depth understanding"
    },
    {
      icon: <Map className="w-8 h-8" />,
      title: "Intelligent Path Planning",
      desc: "A* global planner with dynamic local avoidance and PID smoothing"
    },
    {
      icon: <Shield className="w-8 h-8" />,
      title: "Safety-First Design",
      desc: "Multi-layer safety: emergency stop, collision prediction, recovery behaviors"
    },
    {
      icon: <Cpu className="w-8 h-8" />,
      title: "Production Architecture",
      desc: "Modular, testable, Docker-ready with full telemetry and monitoring"
    }
  ];

  return (
    <div className="bg-[#0a0c14]">
      {/* Navbar */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-[#0a0c14]/90 backdrop-blur-lg border-b border-white/10">
        <div className="max-w-7xl mx-auto px-8 flex items-center justify-between h-20">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-cyan-400 to-violet-500 flex items-center justify-center">
              <span className="text-black font-bold text-xl">A</span>
            </div>
            <div>
              <div className="font-semibold text-2xl tracking-tighter">AETHERNAV</div>
              <div className="text-[10px] text-white/50 -mt-1">AI INDOOR NAVIGATION</div>
            </div>
          </div>

          <div className="flex items-center gap-8 text-sm">
            <a href="#features" className="hover:text-cyan-400 transition-colors">Features</a>
            <a href="#architecture" className="hover:text-cyan-400 transition-colors">Architecture</a>
            <Link to="/docs" className="hover:text-cyan-400 transition-colors">Docs</Link>
            <Link 
              to="/dashboard" 
              className="px-6 py-2.5 bg-white text-black rounded-full font-medium flex items-center gap-2 hover:bg-white/90 transition-all active:scale-[0.985]"
            >
              Launch Dashboard <ArrowRight className="w-4 h-4" />
            </Link>
          </div>
        </div>
      </nav>

      {/* Hero */}
      <div className="pt-20 min-h-[100dvh] flex items-center relative overflow-hidden">
        <div className="absolute inset-0 bg-[radial-gradient(#1a1f2e_0.8px,transparent_1px)] bg-[length:4px_4px]" />
        
        <div className="max-w-5xl mx-auto px-8 relative z-10">
          <div className="max-w-3xl">
            <div className="inline-flex items-center gap-2 px-4 py-1 rounded-full bg-white/5 border border-white/10 text-sm mb-6">
              <div className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse" />
              Production-Ready • Internship Showcase Quality
            </div>

            <h1 className="text-7xl font-semibold tracking-tighter leading-none mb-6">
              Autonomous Indoor<br />Navigation,<br />Reimagined.
            </h1>
            
            <p className="text-2xl text-white/70 max-w-lg mb-10">
              Advanced AI perception + control loop for drones and service robots in complex indoor environments.
            </p>

            <div className="flex items-center gap-4">
              <Link 
                to="/dashboard" 
                className="group px-10 py-4 bg-white text-black rounded-2xl font-semibold flex items-center gap-3 text-lg hover:bg-white/90 transition-all active:scale-[0.985]"
              >
                Open Live Demo
                <ArrowRight className="group-hover:-rotate-45 transition-transform" />
              </Link>
              
              <a 
                href="#features" 
                className="px-8 py-4 border border-white/30 hover:bg-white/5 rounded-2xl font-medium flex items-center gap-2 text-lg transition-all"
              >
                Explore Features
              </a>
            </div>

            <div className="mt-16 flex items-center gap-8 text-sm text-white/50">
              <div>Built with PyTorch • YOLOv8 • A* • React</div>
              <div className="h-px flex-1 bg-white/10" />
              <div>Task 6 Compliant • Advanced Level</div>
            </div>
          </div>
        </div>

        {/* Floating neural orb */}
        <motion.div 
          animate={{ 
            rotate: 360,
            scale: [1, 1.05, 1]
          }}
          transition={{ duration: 25, repeat: Infinity, ease: "linear" }}
          className="absolute right-[-10%] top-1/3 w-[420px] h-[420px] border border-white/10 rounded-full"
        >
          <div className="absolute inset-4 border border-cyan-400/30 rounded-full" />
          <div className="absolute inset-12 border border-violet-400/20 rounded-full" />
        </motion.div>
      </div>

      {/* Features */}
      <div id="features" className="max-w-7xl mx-auto px-8 pb-24">
        <div className="text-center mb-16">
          <div className="text-cyan-400 text-sm tracking-[3px] font-medium">CAPABILITIES</div>
          <h2 className="text-5xl font-semibold tracking-tight mt-3">Built for Real Missions</h2>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          {features.map((feature, index) => (
            <motion.div 
              key={index}
              initial={{ opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="group p-9 bg-white/[0.025] border border-white/10 rounded-3xl hover:border-white/20 transition-all"
            >
              <div className="text-cyan-400 mb-6">{feature.icon}</div>
              <h3 className="text-3xl font-semibold tracking-tight mb-4">{feature.title}</h3>
              <p className="text-lg text-white/70 leading-relaxed">{feature.desc}</p>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Architecture Teaser */}
      <div id="architecture" className="bg-black/40 py-24 border-y border-white/10">
        <div className="max-w-5xl mx-auto px-8">
          <div className="grid md:grid-cols-5 gap-8 items-center">
            <div className="md:col-span-2">
              <div className="uppercase tracking-[2px] text-xs text-white/50 mb-3">SYSTEM DESIGN</div>
              <h3 className="text-5xl font-semibold tracking-tight leading-none">Clean.<br />Modular.<br />Scalable.</h3>
            </div>
            <div className="md:col-span-3 text-lg text-white/70 space-y-6">
              <p>Perception, Navigation, Control, and Telemetry are fully decoupled. Swap YOLO for any detector, A* for RRT*, or add ROS2 nodes with zero core changes.</p>
              <p className="text-white/50">The simulation engine provides pixel-perfect sensor data for rapid iteration before sim-to-real deployment.</p>
            </div>
          </div>
        </div>
      </div>

      {/* CTA */}
      <div className="py-24 text-center">
        <div className="max-w-md mx-auto px-8">
          <h2 className="text-4xl font-semibold tracking-tight mb-4">Ready to explore the future of indoor autonomy?</h2>
          <p className="text-white/60 mb-8">Launch the live dashboard and watch the AI navigate in real time.</p>
          
          <Link 
            to="/dashboard" 
            className="inline-flex items-center gap-3 px-12 py-4 bg-gradient-to-r from-cyan-400 to-violet-500 text-black font-semibold rounded-2xl text-lg hover:brightness-105 active:scale-[0.985] transition-all"
          >
            Enter the Dashboard <Zap className="w-5 h-5" />
          </Link>
        </div>
      </div>

      <footer className="border-t border-white/10 py-12 text-center text-white/40 text-sm">
        AetherNav • Engineered for excellence • 2026
      </footer>
    </div>
  );
};

export default LandingPage;
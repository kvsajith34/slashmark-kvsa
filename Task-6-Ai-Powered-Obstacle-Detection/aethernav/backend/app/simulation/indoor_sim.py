"""
AetherNav - High-Fidelity Indoor Simulation Engine
Pygame-based realistic 2D top-down + pseudo-3D first-person view
with physics, dynamic obstacles, sensor simulation, and collision.

This replaces heavy AirSim for rapid development while remaining
fully compatible with real AirSim/Gazebo via adapter pattern.
"""

import pygame
import numpy as np
import math
import random
import time
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field

# Colors
DARK_BG = (15, 18, 25)
WALL_COLOR = (70, 75, 85)
FLOOR_COLOR = (45, 50, 60)
DRONE_COLOR = (0, 200, 255)
OBSTACLE_COLORS = {
    'wall': (80, 85, 95),
    'chair': (180, 140, 90),
    'table': (160, 120, 80),
    'person': (220, 80, 80),
    'door': (120, 180, 120),
    'dynamic': (255, 180, 50)
}
GRID_COLOR = (35, 40, 50)
PATH_COLOR = (100, 200, 255)
TEXT_COLOR = (230, 235, 245)

@dataclass
class Obstacle:
    x: float
    y: float
    radius: float
    obs_type: str
    vx: float = 0.0
    vy: float = 0.0
    id: int = field(default_factory=lambda: random.randint(1000, 9999))

@dataclass
class DroneState:
    x: float = 2.0
    y: float = 4.5
    yaw: float = 0.0          # radians
    vx: float = 0.0
    vy: float = 0.0
    speed: float = 0.0

class IndoorSimulator:
    def __init__(self, config: dict):
        pygame.init()
        self.config = config
        self.width = int(config['room_width'])
        self.height = int(config['room_height'])
        self.scale = 60  # pixels per meter
        self.screen_width = self.width * self.scale
        self.screen_height = self.height * self.scale

        self.screen = pygame.display.set_mode((self.screen_width + 420, self.screen_height))
        pygame.display.set_caption("AetherNav • Indoor AI Simulator")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 18)
        self.small_font = pygame.font.SysFont("Arial", 14)

        # World
        self.drone = DroneState(
            x=config['drone_start_pos'][0],
            y=config['drone_start_pos'][1],
            yaw=config['drone_start_yaw']
        )
        self.obstacles: List[Obstacle] = []
        self._generate_environment()

        # Simulation state
        self.running = True
        self.paused = False
        self.step_count = 0
        self.last_time = time.time()
        self.fps = 0

        # Perception outputs (simulated)
        self.detected_obstacles: List[Dict] = []
        self.depth_map: Optional[np.ndarray] = None
        self.front_view: Optional[np.ndarray] = None

        # Navigation
        self.planned_path: List[Tuple[float, float]] = []
        self.target: Optional[Tuple[float, float]] = (11.0, 7.5)

        # Metrics
        self.collision_count = 0
        self.distance_traveled = 0.0
        self.last_pos = (self.drone.x, self.drone.y)

    def _generate_environment(self):
        """Generate realistic indoor layout with walls and furniture"""
        self.obstacles = []
        
        # Outer walls
        wall_thickness = 0.25
        self.obstacles.append(Obstacle(0, self.height/2, wall_thickness, 'wall'))
        self.obstacles.append(Obstacle(self.width, self.height/2, wall_thickness, 'wall'))
        self.obstacles.append(Obstacle(self.width/2, 0, wall_thickness, 'wall'))
        self.obstacles.append(Obstacle(self.width/2, self.height, wall_thickness, 'wall'))

        # Internal walls (rooms)
        self.obstacles.append(Obstacle(5.5, 3.0, 0.15, 'wall'))
        self.obstacles.append(Obstacle(8.5, 6.0, 0.15, 'wall'))
        
        # Furniture
        for _ in range(self.config['num_static_obstacles']):
            x = random.uniform(1.5, self.width - 1.5)
            y = random.uniform(1.5, self.height - 1.5)
            r = random.uniform(0.35, 0.7)
            obs_type = random.choice(['chair', 'table', 'door'])
            self.obstacles.append(Obstacle(x, y, r, obs_type))

        # Dynamic obstacles (people, robots)
        for i in range(self.config['num_dynamic_obstacles']):
            x = random.uniform(3, self.width - 3)
            y = random.uniform(3, self.height - 3)
            r = 0.35
            vx = random.uniform(-0.8, 0.8)
            vy = random.uniform(-0.8, 0.8)
            self.obstacles.append(Obstacle(x, y, r, 'dynamic', vx, vy, id=9000+i))

    def update(self, action: Optional[Dict] = None):
        """Main simulation step. Action can come from AI controller."""
        if self.paused:
            return

        dt = 1.0 / self.config['fps']
        self.step_count += 1

        # Apply control action from AI
        if action:
            self._apply_action(action, dt)

        # Update drone physics (simple drag + velocity)
        self.drone.x += self.drone.vx * dt
        self.drone.y += self.drone.vy * dt
        self.drone.speed = math.hypot(self.drone.vx, self.drone.vy)

        # Boundary clamping
        self.drone.x = max(0.6, min(self.drone.x, self.width - 0.6))
        self.drone.y = max(0.6, min(self.drone.y, self.height - 0.6))

        # Update dynamic obstacles
        for obs in self.obstacles:
            if obs.obs_type == 'dynamic':
                obs.x += obs.vx * dt * 0.7
                obs.y += obs.vy * dt * 0.7
                
                # Bounce off walls
                if obs.x < 1 or obs.x > self.width - 1:
                    obs.vx *= -1
                if obs.y < 1 or obs.y > self.height - 1:
                    obs.vy *= -1

        # Collision detection
        self._check_collisions()

        # Update distance traveled
        new_pos = (self.drone.x, self.drone.y)
        self.distance_traveled += math.hypot(new_pos[0] - self.last_pos[0], new_pos[1] - self.last_pos[1])
        self.last_pos = new_pos

        # Simulate perception (this is where real CV would run)
        self._simulate_perception()

        # Update FPS
        now = time.time()
        if now - self.last_time > 1.0:
            self.fps = self.step_count / (now - self.last_time)
            self.step_count = 0
            self.last_time = now

    def _apply_action(self, action: Dict, dt: float):
        """Apply AI decision: {'linear': 0.8, 'angular': 0.3} or similar"""
        linear = action.get('linear', 0.0)
        angular = action.get('angular', 0.0)

        # Update yaw
        self.drone.yaw += angular * dt * 1.5
        self.drone.yaw = (self.drone.yaw + math.pi) % (2 * math.pi) - math.pi

        # Update velocity in drone frame
        speed = linear * self.config.get('max_linear_speed', 1.2)
        self.drone.vx = speed * math.cos(self.drone.yaw)
        self.drone.vy = speed * math.sin(self.drone.yaw)

    def _check_collisions(self):
        for obs in self.obstacles:
            dist = math.hypot(self.drone.x - obs.x, self.drone.y - obs.y)
            if dist < (0.45 + obs.radius):
                self.collision_count += 1
                # Simple bounce back
                dx = self.drone.x - obs.x
                dy = self.drone.y - obs.y
                norm = max(dist, 0.001)
                self.drone.x += (dx / norm) * 0.8
                self.drone.y += (dy / norm) * 0.8
                self.drone.vx *= -0.3
                self.drone.vy *= -0.3

    def _simulate_perception(self):
        """Simulate realistic perception pipeline outputs"""
        self.detected_obstacles = []
        
        # Ray casting for "front camera" detections (simulates LiDAR + vision)
        num_rays = 72
        fov = math.radians(120)
        max_range = 6.0
        
        for i in range(num_rays):
            angle = self.drone.yaw - fov/2 + (i / num_rays) * fov
            ray_x = self.drone.x + max_range * math.cos(angle)
            ray_y = self.drone.y + max_range * math.sin(angle)
            
            # Find closest obstacle intersection
            min_dist = max_range
            hit_obs = None
            for obs in self.obstacles:
                dist = self._ray_obstacle_intersection(
                    self.drone.x, self.drone.y, angle, obs
                )
                if dist < min_dist:
                    min_dist = dist
                    hit_obs = obs
            
            if hit_obs and min_dist < max_range * 0.95:
                # Add noise
                noisy_dist = min_dist * (1 + random.gauss(0, self.config['sensor_noise']))
                self.detected_obstacles.append({
                    'id': hit_obs.id,
                    'type': hit_obs.obs_type,
                    'distance': round(noisy_dist, 2),
                    'angle': round(math.degrees(angle - self.drone.yaw), 1),
                    'confidence': round(random.uniform(0.72, 0.98), 3),
                    'bbox': self._fake_bbox(hit_obs, noisy_dist)
                })

        # Simple depth map (64x48) for visualization
        self.depth_map = np.random.uniform(0.8, 5.5, (48, 64)).astype(np.float32)
        for det in self.detected_obstacles[:8]:
            cx = int(32 + det['angle'] * 0.4)
            if 0 <= cx < 64:
                self.depth_map[:, max(0, cx-3):min(64, cx+4)] = det['distance']

        # Front view simulation (would be real camera frame in production)
        self.front_view = self._generate_front_view()

    def _ray_obstacle_intersection(self, x0, y0, angle, obs) -> float:
        """Simple circle-ray intersection"""
        dx = obs.x - x0
        dy = obs.y - y0
        dist_to_center = math.hypot(dx, dy)
        if dist_to_center < obs.radius:
            return 0.0
        
        # Project
        dot = dx * math.cos(angle) + dy * math.sin(angle)
        if dot < 0:
            return 999.0
        
        perp_dist = abs(dx * math.sin(angle) - dy * math.cos(angle))
        if perp_dist > obs.radius:
            return 999.0
        
        return dot - math.sqrt(obs.radius**2 - perp_dist**2)

    def _fake_bbox(self, obs, dist):
        """Generate plausible bounding box for dashboard"""
        size = max(20, int(180 / max(dist, 0.5)))
        return {
            'x': random.randint(180, 420),
            'y': random.randint(80, 280),
            'w': size + random.randint(-8, 8),
            'h': int(size * 1.3) + random.randint(-5, 5)
        }

    def _generate_front_view(self) -> np.ndarray:
        """Generate a synthetic first-person view image (numpy array)"""
        # Create a pygame surface to draw on
        surf = pygame.Surface((480, 320))
        
        # Ceiling
        surf.fill((25, 28, 35))
        
        # Floor
        pygame.draw.rect(surf, (40, 45, 55), (0, 160, 480, 160))
        
        # Simple perspective walls (triangular-ish effect)
        pygame.draw.polygon(surf, (55, 60, 70), [(0, 0), (240, 0), (240, 160), (0, 320)])
        pygame.draw.polygon(surf, (55, 60, 70), [(480, 0), (240, 0), (240, 160), (480, 320)])
        
        # Draw detected obstacles as simple shapes
        for det in self.detected_obstacles[:5]:
            cx = 240 + int(det['angle'] * 3.5)
            if 40 < cx < 440:
                dist_factor = max(0.3, 1.0 - det['distance'] / 5.5)
                h = int(90 * dist_factor)
                w = int(55 * dist_factor)
                color = OBSTACLE_COLORS.get(det['type'], (200, 150, 100))
                pygame.draw.rect(surf, color, (cx - w//2, 140 - h, w, h), 0)
        
        # Convert surface to numpy array (height, width, 3)
        img = pygame.surfarray.array3d(surf)
        img = img.swapaxes(0, 1)  # pygame gives (width, height, 3), we want (height, width, 3)
        
        return img

    def get_state(self) -> Dict:
        """Return full state for AI controller and dashboard"""
        return {
            'drone': {
                'x': round(self.drone.x, 2),
                'y': round(self.drone.y, 2),
                'yaw': round(math.degrees(self.drone.yaw), 1),
                'speed': round(self.drone.speed, 2),
                'vx': round(self.drone.vx, 2),
                'vy': round(self.drone.vy, 2)
            },
            'detected_obstacles': self.detected_obstacles,
            'depth_map': self.depth_map.tolist() if self.depth_map is not None else None,
            'front_view_shape': self.front_view.shape if self.front_view is not None else None,
            'planned_path': self.planned_path,
            'metrics': {
                'fps': round(self.fps, 1),
                'collisions': self.collision_count,
                'distance_traveled': round(self.distance_traveled, 1),
                'step': self.step_count
            },
            'target': self.target,
            'timestamp': time.time()
        }

    def draw(self):
        """Render beautiful top-down + side panels"""
        self.screen.fill(DARK_BG)
        
        # Main map area
        map_offset_x = 20
        map_offset_y = 20
        
        # Floor
        pygame.draw.rect(self.screen, FLOOR_COLOR, 
                        (map_offset_x, map_offset_y, self.screen_width, self.screen_height))
        
        # Grid
        for i in range(0, self.width + 1, 2):
            pygame.draw.line(self.screen, GRID_COLOR, 
                           (map_offset_x + i * self.scale, map_offset_y),
                           (map_offset_x + i * self.scale, map_offset_y + self.screen_height), 1)
        for i in range(0, self.height + 1, 2):
            pygame.draw.line(self.screen, GRID_COLOR,
                           (map_offset_x, map_offset_y + i * self.scale),
                           (map_offset_x + self.screen_width, map_offset_y + i * self.scale), 1)

        # Obstacles
        for obs in self.obstacles:
            color = OBSTACLE_COLORS.get(obs.obs_type, (200, 150, 100))
            px = map_offset_x + int(obs.x * self.scale)
            py = map_offset_y + int(obs.y * self.scale)
            pr = int(obs.radius * self.scale)
            pygame.draw.circle(self.screen, color, (px, py), pr)
            if obs.obs_type == 'dynamic':
                pygame.draw.circle(self.screen, (255, 220, 100), (px, py), pr + 3, 2)

        # Planned path
        if len(self.planned_path) > 1:
            points = [(map_offset_x + int(p[0] * self.scale), 
                      map_offset_y + int(p[1] * self.scale)) for p in self.planned_path]
            pygame.draw.lines(self.screen, PATH_COLOR, False, points, 3)

        # Drone
        dx = map_offset_x + int(self.drone.x * self.scale)
        dy = map_offset_y + int(self.drone.y * self.scale)
        # Drone body
        pygame.draw.circle(self.screen, DRONE_COLOR, (dx, dy), 14)
        # Heading indicator
        hx = dx + int(22 * math.cos(self.drone.yaw))
        hy = dy + int(22 * math.sin(self.drone.yaw))
        pygame.draw.line(self.screen, (255, 255, 255), (dx, dy), (hx, hy), 4)
        pygame.draw.circle(self.screen, (255, 50, 50), (hx, hy), 5)

        # Target
        if self.target:
            tx = map_offset_x + int(self.target[0] * self.scale)
            ty = map_offset_y + int(self.target[1] * self.scale)
            pygame.draw.circle(self.screen, (255, 100, 100), (tx, ty), 8, 2)
            pygame.draw.line(self.screen, (255, 100, 100), (tx-6, ty-6), (tx+6, ty+6), 2)
            pygame.draw.line(self.screen, (255, 100, 100), (tx-6, ty+6), (tx+6, ty-6), 2)

        # Right panel - Info + Mini front view
        panel_x = self.screen_width + 40
        panel_y = 30
        
        # Title
        title = self.font.render("AETHERNAV • LIVE SIM", True, (0, 220, 255))
        self.screen.blit(title, (panel_x, panel_y))
        
        # Stats
        stats = [
            f"FPS: {self.fps:.1f}",
            f"Position: ({self.drone.x:.1f}, {self.drone.y:.1f}) m",
            f"Heading: {math.degrees(self.drone.yaw):.0f}°",
            f"Speed: {self.drone.speed:.2f} m/s",
            f"Collisions: {self.collision_count}",
            f"Distance: {self.distance_traveled:.1f} m",
            f"Obstacles Detected: {len(self.detected_obstacles)}"
        ]
        
        for i, stat in enumerate(stats):
            surf = self.small_font.render(stat, True, TEXT_COLOR)
            self.screen.blit(surf, (panel_x, panel_y + 45 + i * 26))
        
        # Mini front view
        if self.front_view is not None:
            view_surf = pygame.surfarray.make_surface(self.front_view.swapaxes(0, 1))
            view_surf = pygame.transform.scale(view_surf, (380, 253))
            self.screen.blit(view_surf, (panel_x, panel_y + 280))
            label = self.small_font.render("FRONT CAMERA (AI VIEW)", True, (180, 200, 220))
            self.screen.blit(label, (panel_x + 80, panel_y + 265))

        # Legend
        legend_y = panel_y + 560
        self.screen.blit(self.small_font.render("LEGEND", True, (200, 210, 220)), (panel_x, legend_y))
        legend_items = [
            (DRONE_COLOR, "Drone"),
            (OBSTACLE_COLORS['dynamic'], "Dynamic"),
            (OBSTACLE_COLORS['chair'], "Furniture"),
            (PATH_COLOR, "Planned Path")
        ]
        for i, (color, name) in enumerate(legend_items):
            pygame.draw.rect(self.screen, color, (panel_x, legend_y + 28 + i*22, 14, 14))
            self.screen.blit(self.small_font.render(name, True, TEXT_COLOR), (panel_x + 22, legend_y + 26 + i*22))

        pygame.display.flip()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self._generate_environment()
                    self.drone = DroneState(
    x=self.config['drone_start_pos'][0],
    y=self.config['drone_start_pos'][1],
    yaw=0
)
                elif event.key == pygame.K_ESCAPE:
                    self.running = False

    def run(self, controller_callback=None):
        """Main loop - can be called with AI controller function"""
        print("🚀 AetherNav Simulation started. Press SPACE to pause, R to reset, ESC to quit.")
        
        while self.running:
            self.handle_events()
            
            action = None
            if controller_callback and not self.paused:
                state = self.get_state()
                action = controller_callback(state)
            
            self.update(action)
            self.draw()
            self.clock.tick(self.config['fps'])
        
        pygame.quit()
        print("Simulation ended.")
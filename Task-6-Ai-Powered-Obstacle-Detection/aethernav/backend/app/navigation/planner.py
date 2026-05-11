"""
AetherNav Navigation Module
A* Global Planner + Local Reactive Avoidance + PID Controller
"""

import numpy as np
import heapq
import math
import time
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

@dataclass
class PIDController:
    kp: float
    ki: float
    kd: float
    prev_error: float = 0.0
    integral: float = 0.0
    
    def update(self, error: float, dt: float = 0.1) -> float:
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output

class AStarPlanner:
    def __init__(self, resolution: float = 0.25, safety_margin: float = 0.8):
        self.resolution = resolution
        self.safety_margin = safety_margin
        
    def plan(self, start: Tuple[float, float], goal: Tuple[float, float], 
             obstacles: List[Dict], room_bounds: Tuple[float, float]) -> List[Tuple[float, float]]:
        """A* on occupancy grid"""
        width, height = room_bounds
        grid_w = int(width / self.resolution)
        grid_h = int(height / self.resolution)
        
        # Build occupancy grid
        grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
        
        for obs in obstacles:
            ox = int(obs['x'] / self.resolution) if 'x' in obs else int(obs.get('center', [0,0])[0] / self.resolution)
            oy = int(obs['y'] / self.resolution) if 'y' in obs else int(obs.get('center', [0,0])[1] / self.resolution)
            r = int((obs.get('radius', 0.5) + self.safety_margin) / self.resolution)
            
            for dx in range(-r, r+1):
                for dy in range(-r, r+1):
                    if dx*dx + dy*dy <= r*r:
                        gx, gy = ox + dx, oy + dy
                        if 0 <= gx < grid_w and 0 <= gy < grid_h:
                            grid[gy, gx] = 1
        
        # A* search
        start_g = (int(start[0] / self.resolution), int(start[1] / self.resolution))
        goal_g = (int(goal[0] / self.resolution), int(goal[1] / self.resolution))
        
        if grid[goal_g[1], goal_g[0]] == 1:
            goal_g = self._find_nearest_free(grid, goal_g)
        
        open_set = []
        heapq.heappush(open_set, (0, start_g))
        came_from = {}
        g_score = {start_g: 0}
        f_score = {start_g: self._heuristic(start_g, goal_g)}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == goal_g:
                return self._reconstruct_path(came_from, current, self.resolution)
            
            for neighbor in self._neighbors(current, grid_w, grid_h):
                if grid[neighbor[1], neighbor[0]] == 1:
                    continue
                    
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor, goal_g)
                    f_score[neighbor] = f
                    heapq.heappush(open_set, (f, neighbor))
        
        # Fallback: straight line
        return [start, goal]
    
    def _heuristic(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])
    
    def _neighbors(self, node, w, h):
        x, y = node
        dirs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        return [(x+dx, y+dy) for dx, dy in dirs if 0 <= x+dx < w and 0 <= y+dy < h]
    
    def _reconstruct_path(self, came_from, current, res):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return [(p[0] * res, p[1] * res) for p in path]
    
    def _find_nearest_free(self, grid, pos):
        for r in range(1, 20):
            for dx in range(-r, r+1):
                for dy in range(-r, r+1):
                    nx, ny = pos[0] + dx, pos[1] + dy
                    if 0 <= nx < grid.shape[1] and 0 <= ny < grid.shape[0] and grid[ny, nx] == 0:
                        return (nx, ny)
        return pos

class NavigationController:
    def __init__(self, config: dict):
        self.config = config
        self.planner = AStarPlanner(
            resolution=config['navigation']['planner_resolution'],
            safety_margin=0.9
        )
        self.pid_linear = PIDController(
            kp=config['navigation']['pid']['kp_linear'],
            ki=config['navigation']['pid']['ki_linear'],
            kd=config['navigation']['pid']['kd_linear']
        )
        self.pid_angular = PIDController(
            kp=config['navigation']['pid']['kp_angular'],
            ki=config['navigation']['pid']['ki_angular'],
            kd=config['navigation']['pid']['kd_angular']
        )
        self.last_plan_time = 0
        self.current_path: List[Tuple[float, float]] = []
        self.safety_distance = config['navigation']['safety_distance']
        
    def compute_action(self, state: Dict) -> Dict:
        """Core decision loop: returns {'linear': float, 'angular': float, 'mode': str}"""
        drone = state['drone']
        obstacles = state.get('detected_obstacles', [])
        target = state.get('target', (11.0, 7.5))
        
        current_pos = (drone['x'], drone['y'])
        current_yaw = math.radians(drone['yaw'])
        
        # Safety check - emergency stop
        min_dist = min([o['distance'] for o in obstacles], default=10.0)
        if min_dist < self.safety_distance:
            return {
                'linear': -0.3 if min_dist < 0.8 else 0.0,
                'angular': 0.0,
                'mode': 'EMERGENCY_STOP',
                'reason': f'Obstacle at {min_dist:.1f}m'
            }
        
        # Replan if needed
        now = time.time()
        if now - self.last_plan_time > self.config['navigation']['replan_interval'] or not self.current_path:
            self.current_path = self.planner.plan(
                current_pos, target, 
                [{'x': o.get('center', [0,0])[0], 'y': o.get('center', [0,0])[1], 'radius': o['distance']*0.4} for o in obstacles],
                (self.config['simulation']['room_width'], self.config['simulation']['room_height'])
            )
            self.last_plan_time = now
        
        if not self.current_path:
            self.current_path = [current_pos, target]
        
        # Follow path
        if len(self.current_path) > 1:
            next_wp = self.current_path[1]
            dx = next_wp[0] - current_pos[0]
            dy = next_wp[1] - current_pos[1]
            desired_yaw = math.atan2(dy, dx)
            
            yaw_error = (desired_yaw - current_yaw + math.pi) % (2*math.pi) - math.pi
            angular = self.pid_angular.update(yaw_error)
            
            dist_to_wp = math.hypot(dx, dy)
            linear = min(1.0, dist_to_wp / 1.5) * self.config['navigation']['max_linear_speed']
            
            # Remove waypoint if close
            if dist_to_wp < 0.6:
                self.current_path.pop(0)
            
            mode = 'FOLLOW_PATH'
        else:
            linear = 0.4
            angular = 0.0
            mode = 'APPROACHING_TARGET'
        
        # Confidence based on obstacle density
        confidence = max(0.4, 1.0 - len(obstacles) * 0.08)
        
        return {
            'linear': round(linear, 2),
            'angular': round(angular, 2),
            'mode': mode,
            'confidence': round(confidence, 2),
            'path_length': len(self.current_path)
        }
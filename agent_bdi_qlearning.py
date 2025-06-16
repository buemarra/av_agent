import carla
import math
import random
import numpy as np
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set
from agents.navigation.global_route_planner import GlobalRoutePlanner

class PIDController:
    """PID Controller for steering."""
    def __init__(self, Kp, Ki, Kd, dt=1.0/60):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.integral = 0
        self.prev_error = 0

    def control(self, error):
        """Compute PID control output."""
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative

@dataclass
class Belief:
    """Representa el conocimiento del agente sobre el entorno"""
    current_position: carla.Location
    environment_map: Dict[Tuple[int, int], str]  # 'free', 'obstacle', 'target'
    q_table: Dict[Tuple[int, int], np.ndarray]
    last_reward: float = 0
    visited_states: Set[Tuple[int, int]] = field(default_factory=set)

@dataclass
class Desire:
    """Objetivos del agente"""
    primary: str = "reach_target"
    secondary: List[str] = field(default_factory=lambda: ["avoid_obstacles", "minimize_steps"])

@dataclass
class Intention:
    """Planes de acción del agente"""
    current_plan: List[str] = field(default_factory=list)
    learning_strategy: str = "q_learning"

class BDIVehicleAgent:
    """Agent that controls a vehicle using BDI architecture with Q-Learning"""
    def __init__(self, world, client):
        self.world = world
        self.client = client
        self.map = world.get_map()
        self.grp = GlobalRoutePlanner(self.map, sampling_resolution=2)
        self.vehicle = None
        self.front_camera_img = np.zeros([600,800,3])
        self.route = []
        self.controller = PIDController(Kp=0.05, Ki=0.001, Kd=0.02)
        
        # Q-Learning parameters
        self.lr = 0.2
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.min_epsilon = 0.05
        self.adaptive_lr = True
        self.episode_count = 0
        
        # BDI components
        self.belief = Belief(
            current_position=None,
            environment_map={},
            q_table=defaultdict(lambda: np.zeros(4))  # 4 actions: UP, DOWN, LEFT, RIGHT
        )
        self.desire = Desire()
        self.intention = Intention()
        
        # Actions for Q-Learning: Des
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.action_effects = {
            'UP': (0, 1),
            'DOWN': (0, -1),
            'LEFT': (-1, 0),
            'RIGHT': (1, 0)
        }

    def spawn_vehicle(self):
        """Spawns a vehicle at a fixed or random location."""
        spawn_points = self.map.get_spawn_points()
        use_random_points = False  # Set True for random spawn locations

        if use_random_points:
            a = random.choice(spawn_points).location
            b = random.choice(spawn_points).location
        else:
            a = spawn_points[50].location
            b = spawn_points[100].location

        self.route = self.grp.trace_route(a, b)

        waypoint = self.map.get_waypoint(a)
        vehicle_transform = carla.Transform(
            carla.Location(waypoint.transform.location.x, waypoint.transform.location.y, 2),
            waypoint.transform.rotation
        )

        self.blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = random.choice(self.blueprint_library.filter('vehicle.*'))
        self.vehicle = self.world.spawn_actor(vehicle_bp, vehicle_transform)

        # Initialize belief with current position
        self.belief.current_position = self.vehicle.get_location()
        
        # Draw the planned route
        for i, w in enumerate(self.route):
            color = carla.Color(r=255, g=0, b=0) if i % 10 == 0 else carla.Color(r=0, g=0, b=255)
            self.world.debug.draw_string(w[0].transform.location, 'O', draw_shadow=False, color=color, life_time=120.0, persistent_lines=True)

    def spawn_camera(self):
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(800))
        camera_bp.set_attribute('image_size_y', str(600))
        camera_bp.set_attribute('fov', str(90))
        camera_bp.set_attribute('sensor_tick', str(1/10))    
        transform = carla.Transform(carla.Location(x=0.0, z=1.65))
        self.camera = self.world.spawn_actor(camera_bp, transform, attach_to=self.vehicle)

    def camera_callback(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.copy(array)
        array = np.reshape(array, (image.height, image.width, 4))
        self.front_camera_img = array[:, :, :3]

    def update_spectator(self):
        """Moves the spectator to follow the vehicle."""
        transform = self.vehicle.get_transform()
        location = transform.location
        yaw = transform.rotation.yaw

        dist_camera = 15
        x_offset = math.cos(math.radians(yaw) + math.pi) * dist_camera
        y_offset = math.sin(math.radians(yaw) + math.pi) * dist_camera

        spectator_transform = carla.Transform(
            carla.Location(x=location.x + x_offset, y=location.y + y_offset, z=10),
            carla.Rotation(yaw=yaw, pitch=-15)
        )
        self.world.get_spectator().set_transform(spectator_transform)

        cv2.namedWindow('Real', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Real', self.front_camera_img)
        cv2.waitKey(1)

    def update_beliefs(self, new_position, reward):
        """Actualiza las creencias del agente"""
        self.belief.current_position = new_position
        self.belief.last_reward = reward
        grid_pos = self._location_to_grid(new_position)
        self.belief.visited_states.add(grid_pos)
        
    def _location_to_grid(self, location):
        """Convierte una ubicación CARLA a coordenadas de grid simplificadas"""
        return (int(location.x), int(location.y))
        
    def deliberate(self):
        """Toma decisiones basadas en creencias y deseos usando Q-Learning"""
        current_grid = self._location_to_grid(self.vehicle.get_location())
        
        if self._is_at_target():
            self.desire.primary = "target_reached"
            return "STOP"
            
        # Selección de acción ε-greedy
        if random.random() < self.epsilon:
            return self._explore()
        return self._exploit()
        
    def _explore(self):
        """Exploración inteligente que evita obstáculos"""
        valid_actions = []
        current_pos = self._location_to_grid(self.vehicle.get_location())
        
        for action in self.actions:
            dx, dy = self.action_effects[action]
            new_grid = (current_pos[0] + dx, current_pos[1] + dy)
            if self._is_valid_grid_position(new_grid):
                valid_actions.append(action)
        return random.choice(valid_actions) if valid_actions else random.choice(self.actions)
        
    def _exploit(self):
        """Explotación del conocimiento aprendido"""
        current_grid = self._location_to_grid(self.vehicle.get_location())
        return self.actions[np.argmax(self.belief.q_table[current_grid])]
        
    def _is_valid_grid_position(self, grid_pos):
        """Verifica si una posición de grid es válida"""
        # Implementar lógica para verificar obstáculos en el grid
        return True
        
    def _is_at_target(self):
        """Verifica si el vehículo ha llegado al objetivo"""
        current_pos = self.vehicle.get_location()
        target_pos = self.route[-1][0].transform.location if self.route else None
        if not target_pos:
            return False
            
        distance = math.sqrt((current_pos.x - target_pos.x)**2 + 
                           (current_pos.y - target_pos.y)**2)
        return distance < 2.0  # 2 metros de tolerancia
        
    def update_q_table(self, state, action, reward, next_state):
        """Actualiza la Q-table usando el algoritmo Q-learning"""
        action_idx = self.actions.index(action)
        current_q = self.belief.q_table[state][action_idx]
        max_next_q = np.max(self.belief.q_table[next_state])
        
        # Learning rate adaptativo
        effective_lr = self.lr
        if self.adaptive_lr:
            effective_lr = self.lr / (1 + self.episode_count * 0.001)
            
        new_q = current_q + effective_lr * (reward + self.gamma * max_next_q - current_q)
        self.belief.q_table[state][action_idx] = new_q
        
        # Decaimiento de epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
    def follow_route(self):
        """Moves the vehicle along the planned route using BDI and Q-Learning"""
        if not self.route or not self.vehicle:
            return
            
        current_pos = self.vehicle.get_location()
        current_grid = self._location_to_grid(current_pos)
        
        # Get action from BDI deliberation
        action = self.deliberate()
        
        if action == "STOP":
            self.vehicle.apply_control(carla.VehicleControl(throttle=0, steer=0, brake=1))
            return
            
        # Simulate next state based on action
        dx, dy = self.action_effects[action]
        next_grid = (current_grid[0] + dx, current_grid[1] + dy)
        
        # Calculate reward (simplified for CARLA)
        reward = self._calculate_reward(current_pos, action)
        
        # Update Q-table
        self.update_q_table(current_grid, action, reward, next_grid)
        
        # Update beliefs
        self.update_beliefs(current_pos, reward)
        
        # Convert Q-Learning action to vehicle control
        steer_correction = self._action_to_steering(action)
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=steer_correction, brake=0))
        
    def _calculate_reward(self, current_pos, action):
        """Calcula la recompensa para el aprendizaje por refuerzo"""
        target_pos = self.route[-1][0].transform.location if self.route else None
        if not target_pos:
            return 0
            
        prev_dist = math.sqrt((current_pos.x - target_pos.x)**2 + 
                            (current_pos.y - target_pos.y)**2)
        
        # Simulate next position based on action
        dx, dy = self.action_effects[action]
        new_pos = carla.Location(
            x=current_pos.x + dx,
            y=current_pos.y + dy,
            z=current_pos.z
        )
        
        new_dist = math.sqrt((new_pos.x - target_pos.x)**2 + 
                           (new_pos.y - target_pos.y)**2)
        
        # Basic reward components
        directional_reward = (prev_dist - new_dist) * 15
        step_penalty = -1
        
        # Bonus for moving toward target
        optimal_x_move = np.sign(target_pos.x - current_pos.x)
        optimal_y_move = np.sign(target_pos.y - current_pos.y)
        
        if (dx == optimal_x_move and dy == 0) or (dy == optimal_y_move and dx == 0):
            directional_reward += 2
            
        return directional_reward + step_penalty
        
    def _action_to_steering(self, action):
        """Convierte una acción de Q-Learning a un ángulo de dirección"""
        if action == 'LEFT':
            return -0.5
        elif action == 'RIGHT':
            return 0.5
        return 0  # No steering for UP/DOWN (throttle only)
        
    def run(self):
        """Main loop for vehicle movement."""
        try:
            while True:
                self.follow_route()
                self.update_spectator()
                self.world.tick()
                time.sleep(1.0 / 60)
        except KeyboardInterrupt:
            print("Exiting gracefully...")
            self.vehicle.destroy()

# ===== MAIN SCRIPT =====
if __name__ == "__main__":
    client = carla.Client("localhost", 2000)
    client.set_timeout(10)
    world = client.load_world('Town01')
    # Set synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / 60
    world.apply_settings(settings)
    # Create agent and run
    agent = BDIVehicleAgent(world, client)
    agent.spawn_vehicle()
    agent.spawn_camera()
    agent.camera.listen(lambda image: agent.camera_callback(image))
    agent.run()

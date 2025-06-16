import random
import numpy as np
import math
from collections import defaultdict
import carla
import time
from pyproj import Proj, transform
import asyncio

class CoordinateTransformer:
    """
    Clase para transformar entre coordenadas del simulador CARLA y coordenadas geográficas (lat/lon).
    También incluye métodos para discretizar el espacio para el algoritmo Q-Learning.
    """
    
    def __init__(self, map_origin_latlon=(40.0, -4.0), map_size_meters=(2000, 2000)):
        """
        Inicializa el transformador con:
        - map_origin_latlon: punto de referencia (lat, lon) para la esquina (0,0) del mapa CARLA
        - map_size_meters: tamaño del mapa en metros (ancho, alto)
        """
        self.map_origin_lat, self.map_origin_lon = map_origin_latlon
        self.map_width, self.map_height = map_size_meters
        
        # Proyección UTM (aproximación para pequeñas áreas)
        self.utm_zone = str(math.floor((self.map_origin_lon + 180)/6) + 'N')
        self.proj_utm = Proj(proj='utm', zone=self.utm_zone, ellps='WGS84')
        self.proj_latlon = Proj(proj='latlong', ellps='WGS84')
        
        # Convertir origen a UTM
        self.origin_x, self.origin_y = transform(self.proj_latlon, self.proj_utm, 
                                                self.map_origin_lon, self.map_origin_lat)
    
    def carla_to_latlon(self, x, y):
        """
        Convierte coordenadas CARLA (metros) a latitud/longitud.
        """
        # Convertir a coordenadas UTM (asumiendo que CARLA usa metros)
        utm_x = self.origin_x + x
        utm_y = self.origin_y + y
        
        # Convertir UTM a lat/lon
        lon, lat = transform(self.proj_utm, self.proj_latlon, utm_x, utm_y)
        
        return lat, lon
    
    def latlon_to_carla(self, lat, lon):
        """
        Convierte latitud/longitud a coordenadas CARLA (metros).
        """
        # Convertir lat/lon a UTM
        utm_x, utm_y = transform(self.proj_latlon, self.proj_utm, lon, lat)
        
        # Convertir UTM a coordenadas CARLA
        x = utm_x - self.origin_x
        y = utm_y - self.origin_y
        
        return x, y
    
    def discretize_position(self, x, y, grid_size=(10, 10)):
        """
        Discretiza las coordenadas CARLA para el Q-Learning.
        grid_size: número de celdas en (ancho, alto)
        """
        cell_x = int(np.clip(x / self.map_width * grid_size[0], 0, grid_size[0]-1))
        cell_y = int(np.clip(y / self.map_height * grid_size[1], 0, grid_size[1]-1))
        return (cell_x, cell_y)
    
    def continuous_position(self, cell_x, cell_y, grid_size=(10, 10)):
        """
        Convierte coordenadas discretas de vuelta a posición continua aproximada.
        """
        x = (cell_x + 0.5) * (self.map_width / grid_size[0])
        y = (cell_y + 0.5) * (self.map_height / grid_size[1])
        return x, y


class VehicleEnvironment:
    """Entorno adaptado para vehículos en CARLA"""
    def __init__(self, width=20, height=20, target=(15,15), obstacles=None):
        self.width = width
        self.height = height
        self.target = target
        self.obstacles = set(obstacles) if obstacles else set()
        
        # Acciones más complejas para un vehículo
        self.actions = ['FORWARD', 'FORWARD_LEFT', 'FORWARD_RIGHT', 
                       'BACKWARD', 'BACKWARD_LEFT', 'BACKWARD_RIGHT',
                       'HARD_LEFT', 'HARD_RIGHT', 'STOP']
        
        # Efectos de cada acción (dx, dy)
        self.action_effects = {
            'FORWARD': (1, 0),
            'FORWARD_LEFT': (1, -1),
            'FORWARD_RIGHT': (1, 1),
            'BACKWARD': (-1, 0),
            'BACKWARD_LEFT': (-1, -1),
            'BACKWARD_RIGHT': (-1, 1),
            'HARD_LEFT': (0, -1),
            'HARD_RIGHT': (0, 1),
            'STOP': (0, 0)
        }
        
    def reset(self):
        return (0, 0)
    
    def is_valid_state(self, state):
        x, y = state
        return (0 <= x < self.width and 
                0 <= y < self.height and 
                state not in self.obstacles)
    
    def step(self, state, action):
        dx, dy = self.action_effects[action]
        new_x = max(0, min(self.width-1, state[0] + dx))
        new_y = max(0, min(self.height-1, state[1] + dy))
        new_state = (new_x, new_y)

        if not self.is_valid_state(new_state):
            return new_state, -100, True
        
        if new_state == self.target:
            return new_state, 100, True
        
        # Recompensa basada en distancia + bonus por dirección correcta
        prev_dist = math.sqrt((state[0]-self.target[0])**2 + (state[1]-self.target[1])**2)
        new_dist = math.sqrt((new_x-self.target[0])**2 + (new_y-self.target[1])**2)
        
        directional_reward = (prev_dist - new_dist) * 15
        step_penalty = -1
        
        # Bonus por moverse hacia el objetivo
        optimal_x_move = np.sign(self.target[0] - state[0])
        optimal_y_move = np.sign(self.target[1] - state[1])
        
        if (dx == optimal_x_move and dy == 0) or (dy == optimal_y_move and dx == 0):
            directional_reward += 2
        elif (dx == optimal_x_move and dy == optimal_y_move):
            directional_reward += 5
            
        reward = directional_reward + step_penalty
        
        return new_state, reward, False


class QLearningAgent:
    def __init__(self, env, learning_rate=0.2, discount_factor=0.99,
                 exploration_rate=1.0, exploration_decay=0.999,
                 min_epsilon=0.05, adaptive_lr=True):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.min_epsilon = min_epsilon
        self.adaptive_lr = adaptive_lr
        self.q_table = defaultdict(lambda: np.zeros(len(env.actions)))
        self.episode_count = 0
        
    def get_action(self, state, epsilon=None):
        """Selección ε-greedy mejorada que evita obstáculos"""
        epsilon = epsilon if epsilon is not None else self.epsilon
        
        if random.random() < epsilon:
            # Exploración inteligente que evita obstáculos
            valid_actions = []
            for action in self.env.actions:
                dx, dy = self.env.action_effects[action]
                new_state = (state[0] + dx, state[1] + dy)
                if self.env.is_valid_state(new_state):
                    valid_actions.append(action)
            return random.choice(valid_actions) if valid_actions else random.choice(self.env.actions)
        return self.env.actions[np.argmax(self.q_table[state])]

    def update(self, state, action, reward, next_state):
        """Q-learning con learning rate adaptativo"""
        action_idx = self.env.actions.index(action)
        current_q = self.q_table[state][action_idx]
        max_next_q = np.max(self.q_table[next_state])
        
        # Learning rate adaptativo
        effective_lr = self.lr
        if self.adaptive_lr:
            effective_lr = self.lr / (1 + self.episode_count * 0.001)
            
        new_q = current_q + effective_lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action_idx] = new_q
        
        # Decaimiento de epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def train(self, episodes=2000, eval_interval=100):
        """Entrenamiento con evaluación periódica"""
        rewards = []
        
        for episode in range(episodes):
            self.episode_count += 1
            state = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action = self.get_action(state)
                next_state, reward, done = self.env.step(state, action)
                self.update(state, action, reward, next_state)
                state = next_state
                episode_reward += reward
                
            rewards.append(episode_reward)
            
            if (episode + 1) % eval_interval == 0:
                avg_reward = np.mean(rewards[-eval_interval:])
                print(f"Episodio {episode+1}, Recompensa promedio: {avg_reward:.2f}, Epsilon: {self.epsilon:.4f}")

        return rewards

    def get_optimal_path(self, start_state):
        """Camino óptimo con prevención de ciclos y heurística de distancia"""
        path = [start_state]
        state = start_state
        visited = set([state])
        target = self.env.target
        
        for _ in range(self.env.width * self.env.height * 2):
            if state == target:
                break
                
            # Seleccionar acción considerando Q-values y distancia al objetivo
            best_action = None
            best_value = -np.inf
            
            for action in self.env.actions:
                dx, dy = self.env.action_effects[action]
                new_state = (state[0] + dx, state[1] + dy)
                
                if not self.env.is_valid_state(new_state):
                    continue
                    
                action_idx = self.env.actions.index(action)
                q_value = self.q_table[state][action_idx]
                
                # Heurística: favorecer movimientos que acerquen al objetivo
                dist = math.sqrt((new_state[0]-target[0])**2 + (new_state[1]-target[1])**2)
                heuristic = 1 / (1 + dist)
                total_value = q_value + 0.3 * heuristic
                
                if total_value > best_value and new_state not in visited:
                    best_value = total_value
                    best_action = action
                    
            if best_action is None:
                break
                
            dx, dy = self.env.action_effects[best_action]
            new_state = (state[0] + dx, state[1] + dy)
            path.append(new_state)
            visited.add(new_state)
            state = new_state
            
        return path


class VehicleAgent:
    def __init__(self, world, client):
        self.world = world
        self.client = client
        self.map = world.get_map()
        self.vehicle = None
        self.route = []
        
        # Inicializar transformador de coordenadas
        self.coord_transformer = CoordinateTransformer()
        
        # Crear entorno para Q-Learning
        grid_size = (20, 20)  # Tamaño del grid para discretización
        self.qlearning_env = self.create_qlearning_env(grid_size)
        self.qlearning_agent = QLearningAgent(self.qlearning_env)
    
    def spawn_vehicle(self):
        """Spawns a vehicle at a fixed location"""
        spawn_points = self.map.get_spawn_points()
        spawn_point = spawn_points[50]  # Punto de inicio fijo
        
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = random.choice(blueprint_library.filter('vehicle.*'))
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        
        # Definir ruta (punto final fijo)
        target_point = spawn_points[100].location
        self.route = [target_point]
        
    def create_qlearning_env(self, grid_size):
        """Crea un entorno de Q-Learning basado en el mapa actual"""
        # Definir obstáculos (pueden obtenerse del mapa CARLA)
        obstacles = {(2,2), (3,3), (4,4), (1,4), (4,1), (4,5), (5,4)}  # Ejemplo
        
        # Definir target (puede ser el destino de la ruta)
        target_carla = self.route[-1]
        target_x, target_y = target_carla.x, target_carla.y
        target_cell = self.coord_transformer.discretize_position(target_x, target_y, grid_size)
        
        return VehicleEnvironment(width=grid_size[0], height=grid_size[1], 
                                target=target_cell, obstacles=obstacles)
    
    def get_current_cell(self):
        """Obtiene la celda actual del vehículo para Q-Learning"""
        vehicle_loc = self.vehicle.get_location()
        return self.coord_transformer.discretize_position(vehicle_loc.x, vehicle_loc.y)
    
    def follow_route_with_qlearning(self):
        """Sigue la ruta usando decisiones de Q-Learning"""
        if not self.route or not self.vehicle:
            return
            
        current_cell = self.get_current_cell()
        action = self.qlearning_agent.get_action(current_cell)
        
        # Mapear acción de Q-Learning a control del vehículo
        control = self.action_to_control(action)
        self.vehicle.apply_control(control)
    
    def action_to_control(self, action):
        """Convierte una acción de Q-Learning a un control de vehículo CARLA"""
        # Mapeo de acciones a controles
        if action == 'FORWARD':
            return carla.VehicleControl(throttle=0.7, steer=0.0)
        elif action == 'FORWARD_LEFT':
            return carla.VehicleControl(throttle=0.5, steer=-0.3)
        elif action == 'FORWARD_RIGHT':
            return carla.VehicleControl(throttle=0.5, steer=0.3)
        elif action == 'BACKWARD':
            return carla.VehicleControl(throttle=-0.5, steer=0.0)
        elif action == 'BACKWARD_LEFT':
            return carla.VehicleControl(throttle=-0.3, steer=-0.2)
        elif action == 'BACKWARD_RIGHT':
            return carla.VehicleControl(throttle=-0.3, steer=0.2)
        elif action == 'HARD_LEFT':
            return carla.VehicleControl(throttle=0.1, steer=-0.7)
        elif action == 'HARD_RIGHT':
            return carla.VehicleControl(throttle=0.1, steer=0.7)
        else:  # STOP
            return carla.VehicleControl(throttle=0.0, steer=0.0)
    
    def update_spectator(self):
        """Mueve el espectador para seguir al vehículo"""
        if not self.vehicle:
            return
            
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


async def main():
    # Configurar cliente CARLA
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.load_world('Town01')
    
    # Configurar modo síncrono
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0/60.0
    world.apply_settings(settings)
    
    # Crear agente
    agent = VehicleAgent(world, client)
    agent.spawn_vehicle()
    
    # Entrenar el agente Q-Learning (puede hacerse offline)
    print("Entrenando agente Q-Learning...")
    rewards = agent.qlearning_agent.train(episodes=1000)
    
    # Usar el agente entrenado para controlar el vehículo
    try:
        while True:
            agent.follow_route_with_qlearning()
            agent.update_spectator()
            world.tick()
            await asyncio.sleep(1.0/60.0)
    except KeyboardInterrupt:
        print("Deteniendo la simulación...")
    finally:
        if agent.vehicle:
            agent.vehicle.destroy()
        settings.synchronous_mode = False
        world.apply_settings(settings)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Aplicación terminada por el usuario")
    except Exception as e:
        print(f"\n💥 Error crítico: {str(e)}")

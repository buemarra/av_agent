
import os
import sys
import platform

if platform.system() == 'Windows':
    # Ruta absoluta o relativa a la carpeta que contiene 'agents'
    carla_root = 'C:/CARLA_0.9.15/WindowsNoEditor'
    sys.path.append(os.path.join(carla_root, 'PythonAPI'))
    sys.path.append(os.path.join(carla_root, 'PythonAPI', 'examples'))
    sys.path.append(os.path.join(carla_root, 'PythonAPI', 'carla'))



from datetime import datetime
import os
import csv
import carla
import math
import time
import random
import numpy as np
import cv2
from agents.navigation.global_route_planner import GlobalRoutePlanner


# Constantantes de ajustes de conducción
_delta = 0.5 / 60               # intervalo de tiempo entre cada iteración del lazo de control
_steer_correction = 0.85        # corrección de dirección máxima permitida


class PIDController:
    
    """

        Nombre de la Clase: PIDController   
    
        Descripción corta: env
            Controlador PID para evitar giros buscos
    
        Descripción: evitar vibraciones o giros bruscos (bandazos).
            
            1. Kp (Proporcional)
                Aumenta la respuesta del vehículo.
                Si es muy alto → el vehículo oscila o sobrecorrige.
                Recomendación: empieza con un valor bajo (ej. 0.05) y aumenta gradualmente.
            2. Ki (Integral)
                Corrige errores acumulados (por ejemplo, si el coche siempre gira un poco menos).
                Si es muy alto → puede causar inestabilidad.
                Recomendación: mantenlo bajo (0.001 o incluso 0 al principio).
            3. Kd (Derivativo)
                Suaviza la respuesta, reduce oscilaciones.
                Si es muy alto → puede hacer que el sistema reaccione lentamente.
                Recomendación: empieza con un valor moderado (0.02) y ajusta según el comportamiento.
       
    
    """
    def __init__(self, Kp=0.06, Ki=0.001, Kd=0.02, dt=1.0/60):
        
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


class VehicleAgent:
    
    """Agent that controls a vehicle to follow waypoints."""
    def __init__(self, world, client):
        
        if world != None and client != None:
            self.world = world
            self.client = client
            self.map = world.get_map()
            self.grp = GlobalRoutePlanner(self.map, sampling_resolution=2)
            self.vehicle = None
        
        self.front_camera_img = np.zeros([600,800,3])
        self.route = []

        self.yaw_error = 0.0
        self.steer_correction = 0.0
        self.throttle = 0.5
        self.brake = 0
        self.target_location_y = 0.0
        self.target_location_x = 0.0
        self.vehicle_location_y = 0.0
        self.vehicle_location_x = 0.0

         # self.controller = PIDController(Kp=0.05, Ki=0.001, Kd=0.02) # HERE: Any controller can be implemented
        self.controller = PIDController(Kp=0.06, Ki=00, Kd=0.01) # HERE: Any controller can be implemented

        
        log_file = 'pid_log.csv'
        file_exists = os.path.isfile(log_file)
        with open(log_file, mode='a', newline='') as csvfile:
            fieldnames = ['timestamp', 'Kp', 'Ki', 'Kd', 'yaw_error', 'steer_correction', 'throttle', 'brake',
                                  'target_y', 'vehicle_y', 'target_x', 'vehicle_x']
            self.writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                self.writer.writeheader()
        
        print (f"Fichero csv creado {log_file} ")
        self._updatestate()
        
       
        

    def spawn_vehicle(self):
        
        """
            DESCRIPCION: Muestra o posiciona un vehículo en el mundo de virtual de CARLA

            
        """
        spawn_points = self.map.get_spawn_points()     # Muestra todos los puntos de spawn disponibles en el mapa
        use_random_points = False                      # Deshabilita puntos de spawn aleatorios a False

        if use_random_points:
            a = random.choice(spawn_points).location
            b = random.choice(spawn_points).location
        else:                                           # En esta PoC se utiliza un punto de spawn fijo
            a = spawn_points[50].location               # Waypoint de inicio
            b = spawn_points[100].location              # Waypoint de destino

        self.route = self.grp.trace_route(a, b)         # Traza la ruta

        waypoint = self.map.get_waypoint(a)             # Obtiene el waypoint más cercano al punto de inicio       
        vehicle_transform = carla.Transform(            
            carla.Location(waypoint.transform.location.x, waypoint.transform.location.y, 2),
            waypoint.transform.rotation
        )                                               # Transforma la ubicación del vehículo al waypoint más cercano


        # Spawn the vehicle at the transform: Modificar código de librerias
        self.blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = random.choice(self.blueprint_library.filter('vehicle.tesla.model3'))
        self.vehicle = self.world.spawn_actor(vehicle_bp, vehicle_transform)

      
        for i, w in enumerate(self.route):
            color = carla.Color(r=255, g=0, b=0) if i % 10 == 0 else carla.Color(r=0, g=0, b=255)
            self.world.debug.draw_string(w[0].transform.location, 'O', draw_shadow=False, color=color, life_time=120.0, persistent_lines=True)

    
    def spawn_camera(self):
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        # Modify the attributes of the blueprint to set image resolution and field of view.
        camera_bp.set_attribute('image_size_x', str(800))
        camera_bp.set_attribute('image_size_y', str(600))
        camera_bp.set_attribute('fov', str(90))  # H FOV for ZED with HD720 resolution
        ### By default sensor_tick is 0.00 that means as fast as possible.
        camera_bp.set_attribute('sensor_tick', str(1/10))    
        # Provide the position of the sensor relative to the vehicle.
        transform = carla.Transform(carla.Location(x=0.0, z=1.65))
        # Tell the world to spawn the sensor, don't forget to attach it to your vehicle actor.
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

    def follow_route(self):
        """Moves the vehicle along the planned route using a controller."""
        if not self.route or not self.vehicle:
            return _delta

        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        vehicle_yaw = math.radians(vehicle_transform.rotation.yaw)

        # Find the closest waypoint
        closest_wp = min(self.route, key=lambda wp: wp[0].transform.location.distance(vehicle_location))
        target_location = closest_wp[0].transform.location


        # Actualizacion de atributos posicionamiento

        self.target_location_y = target_location.y
        self.target_location_y = target_location.x
        self.vehicle_location_y = vehicle_location.y
        self.vehicle_location_x = vehicle_location.x


        # Compute heading error
        desired_yaw = math.atan2(target_location.y - vehicle_location.y, target_location.x - vehicle_location.x)
        yaw_error = desired_yaw - vehicle_yaw

        # Normalize yaw error to [-pi, pi]
        yaw_error = (yaw_error + math.pi) % (2 * math.pi) - math.pi

        # Get steering correction from
        self.steer_correction = self.controller.control(yaw_error)
        self.steer_correction = max(min(self.steer_correction, 0.85), -0.85)  # Clamp steering, a vver si da la curva

        # Apply control
        self.vehicle.apply_control(carla.VehicleControl(throttle=self.throttle, steer=self.steer_correction, brake=self.brake))

    def _updatestate(self):

        #print(f"Valores {datetime.now().strftime('%d-%m-%Y %H:%M:%S.%f')[:-3]} {self.controller.Kp}, {self.controller.Ki} {self.controller.Kd} {self.yaw_error} {self.steer_correction} {self.steer_correction} {self.throttle} {self.brake} {self.target_location_y} {self.target_location_x} {self.vehicle_location_x} \n")
        with open('pid_log.csv', mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=[
                'timestamp', 'Kp', 'Ki', 'Kd', 'yaw_error', 'steer_correction',
                'throttle', 'brake', 'target_y', 'vehicle_y', 'target_x', 'vehicle_x'
                ])
            writer.writerow({
                'timestamp': datetime.now().strftime('%d-%m-%Y %H:%M:%S.%f')[:-3],
                'Kp': self.controller.Kp,
                'Ki': self.controller.Ki,
                'Kd': self.controller.Kd,
                'yaw_error': self.yaw_error,
                'steer_correction': self.steer_correction,
                'throttle': self.throttle,
                'brake': self.brake,
                'target_y': self.target_location_y,
                'vehicle_y': self.vehicle_location_y,
                'target_x': self.target_location_x,
                'vehicle_x': self.vehicle_location_x
                })

    def run(self):
        # Prepare log file
                
                    
        """
            DESCRIPCION: Lazo cerrado para el movimiento del vehículo
        """

        
        if platform.system() == 'Linux' and 'Ubuntu' in platform.version():
            # Código exclusivo para Ubuntu
            print(f"Iniciando el Sistema {platform.version()} .... ")

            try:
            
                self._updatestate()
                while True:
                    self.follow_route()
                    self.update_spectator()
                    self.world.tick()
                    time.sleep(_delta)
                
                    self._updatestate()
        
            except KeyboardInterrupt:
                print("Exiting gracefully...")
                self.vehicle.destroy()

        else:
            print(f"Iniciando el Sistema {platform.system()} Version: {platform.version()} ....\n ")
            try:
            
                while True:
                
                    time.sleep(_delta)
                    self._updatestate()
        
            except KeyboardInterrupt:
                print(f"Saliendo del Sistema {platform.system()}.\n ")
                self.vehicle.destroy()




# ===== MAIN SCRIPT =====
def main():

    try:
        client = carla.Client("localhost", 2000)
        client.set_timeout(10)
        world = client.load_world('Town01')
        # Set synchronous mode
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = _delta
        world.apply_settings(settings)
        # Create agent and run
        agent = VehicleAgent(world, client)
        agent.spawn_vehicle()
        agent.spawn_camera()
        agent.camera.listen(lambda image: agent.camera_callback(image))
    
    except RuntimeError as e:
        print(f"[ADVERTENCIA] Error al conectar con CARLA: {e}")
        simulador_conectado = False

    if simulador_conectado == False:
        agent = VehicleAgent(world=None, client=None)
    

    
        
    agent.run()


if __name__ == '__main__':
    main()
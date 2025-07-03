import sys
import os
import carla
import math
import time
import random
import numpy as np
import cv2
import csv
from datetime import datetime
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

class VehicleAgent:
    """Agent that controls a vehicle to follow waypoints."""
    def __init__(self, world, client):
        self.world = world
        self.client = client
        self.map = world.get_map()
        self.grp = GlobalRoutePlanner(self.map, sampling_resolution=2)
        self.vehicle = None
        self.front_camera_img = np.zeros([600, 800, 3])
        self.route = []
        self.controller = PIDController(Kp=0.05, Ki=0.001, Kd=0.02)
        self.yaw_error = 0.0
        self.steer_correction = 0.0
        self.target_location_y = 0.0
        self.target_location_x = 0.0
        self.vehicle_location_y = 0.0
        self.vehicle_location_x = 0.0
        self.error_x = 0.0
        self.error_y = 0.0
        self.throttle = 0.5
        self.brake = 0
        self.delta = 0.5 / 60

        self.veh_length = 0.0
        self.veh_width = 0.0
        self.veh_height = 0.0

        _fecha = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
        self.log_file = f'[{_fecha}]-pid_log.csv'
        
        with open(self.log_file, mode='w', newline='') as csvfile:
            fieldnames = ['timestamp', 'Kp', 'Ki', 'Kd', 'yaw_error', 
                          'steer_correction', 'acelerador', 'brake', 
                          'target_y', 'vehicle_y', 'target_x','vehicle_x',
                          'error_x', 'error_y', 'vehicle_height', 
                          'vehicle_width','vehicle_length']
            # Initialize the CSV file with headers
            self.writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            self.writer.writeheader()

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

        # Draw the planned route
        for i, w in enumerate(self.route):
            color = carla.Color(r=255, g=0, b=0) if i % 10 == 0 else carla.Color(r=0, g=0, b=255)
            self.world.debug.draw_string(w[0].transform.location, 'O', draw_shadow=False, color=color, life_time=120.0, persistent_lines=True)


    def get_vehicle_dimensions(self, world=None):
        '''Obtiene las dimensiones del vehículo en metros: 0.9.14 version'''
        if not self.vehicle:
            return None
        bounding_box = self.vehicle.bounding_box
        extent = bounding_box.extent
        self.veh_length = extent.x * 2
        self.veh_width = extent.y * 2 
        self.veh_height = extent.z * 2
        

        
    
    
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
            return

        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        vehicle_yaw = math.radians(vehicle_transform.rotation.yaw)

        # Find the closest waypoint
        closest_wp = min(self.route, key=lambda wp: wp[0].transform.location.distance(vehicle_location))
        target_location = closest_wp[0].transform.location

        # Compute heading error
        desired_yaw = math.atan2(target_location.y - vehicle_location.y, target_location.x - vehicle_location.x)
        yaw_error = desired_yaw - vehicle_yaw

        # Normalize yaw error to [-pi, pi]
        yaw_error = (yaw_error + math.pi) % (2 * math.pi) - math.pi

        # Get steering correction from
        steer_correction = self.controller.control(yaw_error)
        steer_correction = max(min(steer_correction, 1.0), -1.0)  # Clamp steering

        
        # Actualizacion de atributos posicionamiento

        self.target_location_y = target_location.y
        self.target_location_y = target_location.x
        self.vehicle_location_y = vehicle_location.y
        self.vehicle_location_x = vehicle_location.x
        
        self.steer_correction = steer_correction
        self.yaw_error = yaw_error


        # Errores absolutos de la trayectoria
        self.error_x = target_location.x - vehicle_location.x
        self.error_y = target_location.y - vehicle_location.y
        
        # Apply control
        self.vehicle.apply_control(carla.VehicleControl(throttle=self.throttle, steer= self.steer_correction, brake=self.brake))


    def _updatestate(self):
        with open(self.log_file, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['timestamp', 'Kp', 'Ki', 'Kd', 'yaw_error',
                                                      'steer_correction', 'acelerador', 'brake',
                                                      'target_y', 'vehicle_y', 'target_x', 'vehicle_x','error_x', 'error_y',
                                                      'vehicle_height', 'vehicle_width','vehicle_length'])
            writer.writerow({'timestamp': datetime.now().strftime('%d-%m-%Y %H:%M:%S.%f')[:-3], 
                             'Kp': self.controller.Kp, 'Ki': self.controller.Ki, 'Kd': self.controller.Kd, 'yaw_error': self.yaw_error,
                             'steer_correction': self.steer_correction, 'acelerador': self.throttle, 'brake': self.brake, 'target_y': self.target_location_y, 
                             'vehicle_y': self.vehicle_location_y, 'target_x': self.target_location_x, 'vehicle_x': self.vehicle_location_x,
                             'error_x': self.error_x, 'error_y': self.error_y, 'vehicle_height': self.veh_height, 'vehicle_width': self.veh_width,
                             'vehicle_length': self.veh_length})

    def run(self):
        """Main loop for vehicle movement."""
        try:
            while True:
                self.follow_route()
                self.update_spectator()
                self.world.tick()
                sys.stdout.write(f'[INFO] DELTA: {self.delta} Kp:{self.controller.Kp}, Kd:{self.controller.Kd}, Ki:{self.controller.Ki}, Integral:{self.controller.integral}, Error: {self.controller.prev_error} \n')
                self._updatestate()
                time.sleep(self.delta)
        except KeyboardInterrupt:
            print("Salida controlada...")
            self.vehicle.destroy()

def main():
    _delta = 0.5 / 60
    try:
        print('[1] Iniciando cliente de CARLA\n')
        client = carla.Client("localhost", 2000)
        client.set_timeout(10)
        world = client.load_world('Town01')
        print(f'[2] Cliente CARLA instanciado\n')
        print("[3] Inicializando Settigns\n")  # Set synchronous mode
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = _delta
        world.apply_settings(settings)
        print("[4] Settings inicializados\n") # Create agent and run
        print("[5] Instanciando vehículo en el mundo CARLA\n")
        agent = VehicleAgent(world, client)
        agent.spawn_vehicle()
        agent.spawn_camera()
        agent.camera.listen(lambda image: agent.camera_callback(image))
        agent.delta = _delta
        agent.get_vehicle_dimensions(world= world)
        print(f"[5.1] Dimensiones del vehículo: {agent.veh_length} x {agent.veh_width} x {agent.veh_height}\n")
        print("[6] Vehículo instanciado en el mundo Carla\n")

        # Spawn a walker (pedestrian) in the world
        walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
        walker = world.spawn_actor(walker_controller_bp, carla.Transform())
        print("[7] Caminante no hay camino ..  se hace camino al andar\n")
        print("[8] Iniciando ejecución del agente\n")
        
        
        
        
        agent.run()
        
        
        
        
        
    except Exception as _ex:
        print(f"Error al conectar con CARLA: {_ex}")
    finally:
        print("Finalizando ejecución del agente")


if __name__ == "__main__":
    main()

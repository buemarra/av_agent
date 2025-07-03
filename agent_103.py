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
                          'steer_correction', 'acelerador', 'brake', 'target_y', 
                          'vehicle_y', 'target_x', 'vehicle_x','eror_x', 'error_y', 'vehicle_height', 
                          'vehicle_width','vehicle_length']
            self.writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            self.writer.writeheader()

    def spawn_vehicle(self):
        """Spawns a vehicle at a fixed or random location."""
        spawn_points = self.map.get_spawn_points()
        a = spawn_points[50].location
        vehicle_transform = carla.Transform(
            carla.Location(a.x, a.y, a.z + 2),
            carla.Rotation(pitch=0, yaw=a.rotation.yaw, roll=0)
        )
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = random.choice(blueprint_library.filter('vehicle.*'))
        self.vehicle = self.world.spawn_actor(vehicle_bp, vehicle_transform)

    
    
    def get_vehicle_dimensions(self):
        if not self.vehicle:
            return None
        
        # Obtener el blueprint del vehículo
        blueprint = self.vehicle.get_blueprint()

        # Obtener las dimensiones del vehículo
        extent = blueprint.get_attribute('extent').as_vector3d()

        # Las dimensiones están en metros
        self.veh_length = extent.x
        self.veh_width = extent.y
        self.veh_height = extent.z

        return self.veh_length, self.veh_width, self.veh_height

    
    
    def spawn_camera(self):
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(800))
        camera_bp.set_attribute('image_size_y', str(600))
        camera_bp.set_attribute('fov', str(90))
        camera_bp.set_attribute('sensor_tick', str(1/10))
        transform = carla.Transform(carla.Location(x=0.0, z=1.65))
        camera_sensor = self.world.spawn_actor(camera_bp, transform, attach_to=self.vehicle)
        return camera_sensor

    def camera_callback(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.copy(array)
        array = np.reshape(array, (image.height, image.width, 4))
        self.front_camera_img[:, :, :3] = array[:, :, :3]

    def update_spectator(self):
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
        if not self.route or not self.vehicle:
            return
        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        vehicle_yaw = math.radians(vehicle_transform.rotation.yaw)
        closest_wp = min(self.route, key=lambda wp: wp[0].transform.location.distance(vehicle_location))
        target_location = closest_wp[0].transform.location
        desired_yaw = math.atan2(target_location.y - vehicle_location.y, target_location.x - vehicle_location.x)
        yaw_error = desired_yaw - vehicle_yaw
        yaw_error = (yaw_error + math.pi) % (2 * math.pi) - math.pi
        steer_correction = self.controller.control(yaw_error)
        steer_correction = max(min(steer_correction, 1.0), -1.0)
        self.target_location_y = target_location.y
        self.target_location_x = target_location.x
        self.vehicle_location_y = vehicle_location.y
        self.vehicle_location_x = vehicle_location.x
        self.steer_correction = steer_correction
        self.yaw_error = yaw_error
        self.error_x = target_location.x - vehicle_location.x
        self.error_y = target_location.y - vehicle_location.y
        self.vehicle.apply_control(carla.VehicleControl(throttle=self.throttle, steer=self.steer_correction, brake=self.brake))

    def _updatestate(self):
        with open(self.log_file, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['timestamp', 'Kp', 'Ki', 'Kd', 'yaw_error',
                                                      'steer_correction', 'acelerador', 'brake',
                                                      'target_y', 'vehicle_y', 'target_x', 'vehicle_x','eror_x', 'error_y',
                                                      'vehicle_height', 'vehicle_width','vehicle_length'])
            writer.writerow({'timestamp': datetime.now().strftime('%d-%m-%Y %H:%M:%S.%f')[:-3], 
                             'Kp': self.controller.Kp, 'Ki': self.controller.Ki, 'Kd': self.controller.Kd, 'yaw_error': self.yaw_error,
                             'steer_correction': self.steer_correction, 'acelerador': self.throttle, 'brake': self.brake, 'target_y': self.target_location_y, 
                             'vehicle_y': self.vehicle_location_y, 'target_x': self.target_location_x, 'vehicle_x': self.vehicle_location_x,
                             'eror_x': self.error_x, 'error_y': self.error_y, 'vehicle_height': self.veh_height, 'vehicle_width': self.veh_width,
                             'vehicle_length': self.veh_length})

    def run(self):
        try:
            while True:
                self.follow_route()
                self.update_spectator()
                self.world.tick()
                self._updatestate()
                time.sleep(1.0 / 60)
        except KeyboardInterrupt:
            print("Exiting gracefully...")
            self.vehicle.destroy()

def main():
    _delta = 0.5 / 60
    try:
        client = carla.Client("localhost", 2000)
        client.set_timeout(10)
        world = client.load_world('Town01')
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = _delta
        world.apply_settings(settings)
        agent = VehicleAgent(world, client)
        agent.spawn_vehicle()
        camera_sensor = agent.spawn_camera()
        camera_sensor.listen(lambda image: agent.camera_callback(image))
        agent.run()
    except Exception as _ex:
        print(f"Error al conectar con CARLA: {_ex}")
    finally:
        print("Finalizando ejecución del agente")

if __name__ == "__main__":
    main()

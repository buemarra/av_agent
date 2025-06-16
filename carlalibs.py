import numpy as np
import math
from pyproj import Proj, transform

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
        self.utm_zone = str(math.floor((self.map_origin_lon + 180)/6) + 'N'
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
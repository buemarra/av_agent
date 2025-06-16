import carla

try:
    # Intentar conectar con el servidor CARLA
    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)
    world = client.get_world()
    print("¡Conexión exitosa con CARLA!")
    print(f"Mapa actual: {world.get_map().name}")
except Exception as e:
    print(f"Error al conectar con CARLA: {e}")
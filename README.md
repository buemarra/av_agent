Titulo: Vehículo Autonomo en Entornos Virtuales (CARLA)
Autor: Ramiro Bueno Martínez (MSc Inteligencia Artificial para el sector de la Energía e Infraestructuras)
Fecha: 15/06/2025

Definición del Problema: La presente PoC modela un vehiculo autónomo en el Entorno Virtual del
Simulador Carla (Carla 0.14). El objetivo del presente código desarrollado se centra en establecer
un Sistema de Conciencia Situacional muy básico, que dotaría a un vehículo autonomo de capacidades
para percibir el entorno a través de sensores ( camaras de video), unos funciones básicas para comprender
dicho entorno y tomar decisiones muy básicas, que permitan una proyección del estado deseado
en terminos de seguir una ruta, previamente planificada, anticipandose y eludiendo obstaculos y
situaciones de conducción de dificulten el trayecto.

Descripción: La presente prueba de concepto se sustenta en la teória y arquitectura de Agentes Autonomos
que se les dota de una capa de percepción a través de sensores colocados en el vehículo. Se define
un controlador inicial basado en un control PID sincrono corriendo en el espacio discreto de estados
con una delta temporal de (_delta = 0.5 / 60) segs, cuyo objetivo es seguir la ruta previamente planificada
en termino de way-points con una conducción los mas suave posible adaptandose al escenario de conducción. 
En este caso el escenario de conducción está definido en una ciudad concreta, con un punto inicial de partida
y un punto final de llegada.

Objetivos: Familiarizarse con el entorno virtual CARLA, desarrollar un conjunto de datos (.csv) que permitan
interpretar en termino de variables, atributos y caracteristica y de forma mas simplista de las entradas/salidas
que intervienen en el proceso de conducción de un vehículo, con posterioridad se analizarán y se implementará
un sistema mas complejo basado en marcos de trabajo relacionados con el aprendizaje por refuerzo y/o inferencia activa,
una de las lineas maestras de la investigación doctoral.



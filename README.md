# SLAM Geométrico: Fase 1 - Reconstrucción Dispersa

Brayan Alpizar Elizondo  
Ingenieria en Computadores  
Instituto Tecnológico de Costa Rica (TEC)  
Samsung S24 (Sensor 2x Calibrado)

## 1. Descripcion General
Este proyecto implementa la primera etapa de un sistema SLAM (Simultaneous Localization and Mapping). El software procesa pares de imagenes para estimar el movimiento relativo de una camara en un entorno tridimensional y genera una representacion dispersa del escenario mediante una nube de puntos 3D. El sistema esta diseñado para ser modular, permitiendo la inspeccion detallada de cada etapa del pipeline de vision por computadora.

## 2. Arquitectura de Clases y Modulos

### DataProvider (data_acquisition_provider.py)
Gestiona la entrada de datos del sistema. Su funcion principal es cargar la matriz intrinseca (K) y los coeficientes de distorsion calculados previamente en la fase de calibracion. Tambien se encarga de la lectura de frames, asegurando que el procesamiento posterior cuente con datos geometricamente validos.

### ORBFeatureExtractor (feature_extraction_orb.py)
Encargado de la percepcion visual. Utiliza el algoritmo ORB (Oriented FAST and Rotated BRIEF) para detectar puntos de interes y generar descriptores binarios. Se ha configurado para extraer hasta n caracteristicas por imagen, garantizando una densidad suficiente para la triangulacion en entornos con componentes electronicos pequeños.

### FeatureMatcher (feature_matching_ransac.py)
Establece las correspondencias entre imagenes. Utiliza un buscador de fuerza bruta con norma Hamming y aplica el Test de Ratio de Lowe para filtrar ambigüedades. Incorpora el calculo de la Matriz Esencial bajo el esquema RANSAC para eliminar correspondencias que no cumplen con la restriccion epipolar.

### MotionEstimator (motion_estimation_geometry.py)
Calcula la cinematica de la camara. Descompone la Matriz Esencial en una matriz de rotacion (R) y un vector de traslacion (t). Aplica una verificacion de quiralidad (Cheirality Check) para determinar cual de las cuatro soluciones posibles es la que situa los puntos 3D frente a ambos centros opticos.

### MapTriangulator (map_triangulation_3d.py)
Realiza la reconstruccion espacial. Utiliza las matrices de proyeccion de ambas camaras para intersectar los rayos de luz provenientes de los puntos 2D. La salida es un conjunto de coordenadas euclidianas (X, Y, Z) tras realizar la division perspectiva sobre el espacio homogeneo.

## 3. Instrucciones de Ejecucion
Para ejecutar el pipeline completo y abrir las herramientas de inspeccion, utilice el siguiente comando:

python main.py

## 4. Guia de Visualizacion e Interaccion
El sistema despliega una interfaz doble sincronizada para validar la precision del algoritmo:

### Visor de Correspondencias 2D (OpenCV)
Muestra el flujo optico entre las dos imagenes cargadas. 
- Trackbar (Match ID): Permite desplazarse individualmente por cada match detectado.
- Feedback Visual: El match seleccionado se resalta en verde con circulos azules, permitiendo verificar si los puntos corresponden al mismo componente fisico.

### Inspeccion de Nube de Puntos 3D (Matplotlib)
Genera una grafica interactiva de los landmarks triangulados.
- Interaccion de Puntos: El usuario puede hacer clic en cualquier punto 3D de la grafica.
- Consola de Inspeccion: Al presionar un punto, la terminal imprimira el ID del match y sus coordenadas exactas. Este ID puede ser buscado manualmente en el visor 2D para comparar el error de reproyeccion.

## 5. Limitaciones Tecnicas y Errores Conocidos


1. Ambigüedad de Escala: Al utilizar una sola camara (monocular), el sistema no conoce la escala absoluta. Las distancias son relativas a la unidad, no necesariamente en metros o milimetros reales.
2. Error por Baseline Corto: Si el desplazamiento lateral entre las fotos es muy pequeño, los angulos de triangulacion resultan en una alta incertidumbre de profundidad (Eje Z).
3. Dependencia de Calibracion: Los resultados son altamente sensibles a la matriz K. Se asume que el Samsung S24 entrega imagenes rectificadas por hardware, por lo que una sobre-correccion mediante software podria degradar la geometria.
4. Ruido de Profundidad: Puntos extremadamente lejanos pueden presentar mayor dispersion debido a la limitada resolucion de los descriptores ORB en distancias largas.

## 6. Requisitos del Sistema
- Python 3.10 o superior.
- OpenCV (opencv-python).
- NumPy.
- Matplotlib.


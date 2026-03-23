# Camera Calibration Process

Este documento describe el proceso utilizado para calibrar la cámara utilizada en el proyecto de **Visual SLAM y reconstrucción 3D**.

La calibración permite obtener los **parámetros intrínsecos de la cámara** y los **coeficientes de distorsión**, necesarios para:

* corregir distorsión de lente
* estimar geometría epipolar
* realizar triangulación
* estimar pose de cámara
* construir mapas SLAM

---

# Pipeline de calibración

```
Calibration video
      │
      ▼
src/calibration_process/extract_calibration_frames.py
      │
      ▼
Frames con patrón detectado
(resources/calibration_frames)
      │
      ▼
src/calibration_process/calibration_parameters.py
      │
      ▼
Camera Matrix + Distortion Coefficients
(calibration_results)
      │
      ▼
src/calibration_process/calibration_error.py
      │
      ▼
Reprojection Error
(verificación de calidad)
```

---

# Estructura de carpetas

```
Proyecto Investigacion
│
├── calibration_results
│   ├── camera_matrix.npy
│   └── dist_coeffs.npy
│
├── resources
│   │
│   ├── calibration_video
│   │   └── calibration_video.mp4
│   │
│   ├── calibration_frames
│   │   └── frame_*.jpg
│   │
│   ├── camera_info
│   │
│   └── chessboard
│       └── chessboard_9x6_30mm.png
│
└── src
    └── calibration_process
        ├── extract_calibration_frames.py
        ├── calibration_parameters.py
        └── calibration_error.py
```

---

# Descripción de archivos

## extract_calibration_frames.py

```
src/calibration_process/extract_calibration_frames.py
```

Función:

* abre el video de calibración
* detecta el patrón **chessboard**
* guarda únicamente los frames válidos

Proceso:

```
Video → detectar tablero → guardar frame
```

Salida:

```
resources/calibration_frames/frame_*.jpg
```

---

## calibration_parameters.py

```
src/calibration_process/calibration_parameters.py
```

Función:

* carga los frames de calibración
* detecta esquinas del tablero
* calcula los parámetros de cámara con `cv2.calibrateCamera()`

Parámetros calculados:

```
Camera Matrix (K)
Distortion Coefficients
Rotation Vectors
Translation Vectors
```

Salida:

```
calibration_results/camera_matrix.npy
calibration_results/dist_coeffs.npy
```

---

## calibration_error.py

```
src/calibration_process/calibration_error.py
```

Función:

* calcula el **error de reproyección**

Este error mide qué tan bien el modelo de cámara reproduce las observaciones reales.

Valores típicos:

```
< 0.2   excelente
< 0.5   muy bueno
< 1.0   bueno
> 2.0   mala calibración
```

Resultado obtenido en este proyecto:

```
Reprojection error ≈ 0.134
```

Esto indica **una calibración de muy alta precisión**.

---

# Patrón de calibración

Se utiliza un **tablero de ajedrez** generado por script.

Configuración:

```
Internal corners: 9 x 6
Square size: 30 mm
```

Archivo:

```
resources/chessboard/chessboard_9x6_30mm.png
```

---

# Configuración recomendada de la cámara

Para obtener una calibración confiable se utilizaron las siguientes configuraciones:

```
Dispositivo: Samsung Galaxy S24
Resolución: 1080p
Aspect ratio: 16:9
FPS: 30
Zoom: x1 (sensor principal)
Estabilización: DESACTIVADA
```

Motivos:

### 1080p

Mayor cantidad de información visual para detectar esquinas.

### 16:9

Mantiene proporción consistente con la mayoría de pipelines de visión.

### Sin estabilización

La estabilización digital puede alterar la geometría de la imagen y afectar la calibración.

### Zoom x1

Permite utilizar el **sensor principal del teléfono**, reduciendo distorsión de lente.

---

# Recomendaciones para capturar el video

Duración recomendada:

```
20 – 30 segundos
```

Durante la grabación mover el tablero para cubrir:

```
✔ centro de la imagen
✔ esquinas
✔ diferentes distancias
✔ diferentes orientaciones
```

Ejemplo de poses:

```
     ┌───────────────┐
     │      ███      │
     │               │
     │               │
     └───────────────┘

     ┌───────────────┐
     │               │
     │   ███         │
     │               │
     └───────────────┘
```

Esto mejora la estimación de:

```
focal length
principal point
distortion coefficients
```

---

# Resultado de calibración

Camera Matrix:

```
[[606.53646783   0.         405.46824346]
 [  0.         606.99906883 266.8142911 ]
 [  0.           0.           1.        ]]
```

Distortion coefficients:

```
[ 0.21078517  0.07870492  0.02281009 -0.01257568 -0.72937922 ]
```

Reprojection Error:

```
0.1347
```

---

# Uso de los parámetros de cámara

Estos parámetros se utilizan para **corregir distorsión** antes de ejecutar el pipeline de SLAM.

Ejemplo:

```
undistorted = cv2.undistort(image, K, dist_coeffs)
```

Pipeline posterior:

```
Undistort
   │
   ▼
Feature Detection (ORB)
   │
   ▼
Feature Matching
   │
   ▼
Essential Matrix
   │
   ▼
Camera Pose
   │
   ▼
Triangulation
   │
   ▼
SLAM Map
```

---
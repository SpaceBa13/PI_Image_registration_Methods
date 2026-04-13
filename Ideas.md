------------------------------------------------------------------------

# 1. Arquitectura Planteada

```bash
SLAM-System
│
├── Phase_1_Geometric_SLAM
│   │
│   ├── Data_Acquisition
│   │   ├── Camera_Stream
│   │   │   ├── Video Frames
│   │   │   └── Timestamp Synchronization
│   │   │
│   │   └── Camera_Calibration
│   │       ├── Intrinsic Matrix (K)
│   │       ├── Distortion Parameters
│   │       └── Camera Projection Model
│   │
│   ├── Feature_Extraction
│   │   ├── Keypoint Detection
│   │   │   ├── ORB
│   │   │   ├── SIFT
│   │   │   └── AKAZE
│   │   │
│   │   └── Descriptor Computation
│   │       ├── Binary Descriptors (ORB)
│   │       └── Floating Descriptors (SIFT)
│   │
│   ├── Feature_Matching
│   │   ├── Descriptor Matching
│   │   │   ├── Hamming Distance
│   │   │   └── FLANN Matcher
│   │   │
│   │   ├── Match Filtering
│   │   │   ├── Ratio Test
│   │   │   └── Cross Check
│   │   │
│   │   └── Outlier Removal
│   │       └── RANSAC
│   │
│   ├── Pose_Estimation
│   │   ├── Essential Matrix Estimation
│   │   │   └── cv2.findEssentialMat()
│   │   │
│   │   ├── Camera Motion Recovery
│   │   │   └── cv2.recoverPose()
│   │   │
│   │   └── Relative Motion Vector
│   │       ├── Rotation (R)
│   │       └── Translation (t)
│   │
│   ├── Map_Generation
│   │   ├── Point Triangulation
│   │   │   └── cv2.triangulatePoints()
│   │   │
│   │   ├── 3D Point Cloud Construction
│   │   │   └── Feature Landmarks
│   │   │
│   │   └── Map Update
│   │       ├── Keyframe Selection
│   │       └── Landmark Management
│   │
│   └── Trajectory_Estimation
│       ├── Pose Accumulation
│       └── Camera Path Reconstruction
│
│
├── Phase_2_GPS_Supervised_Training
│
│
└── Phase_3_Corrected_SLAM
```

## 2.SLAM geométrico

El SLAM geométrico es la base del sistema. Aquí la estimación del movimiento y del mapa se realiza únicamente usando propiedades geométricas de las imágenes, sin ningún aprendizaje automático.

El proceso comienza con la adquisición de imágenes de la cámara, normalmente en forma de video. Es importante que la cámara esté calibrada para obtener la matriz intrínseca (K), que describe cómo la cámara proyecta puntos 3D en el plano de imagen. Esta calibración se puede realizar usando herramientas de OpenCV con un patrón de tablero de ajedrez (ya realizada).

Luego se realiza la detección de features, que son puntos distintivos de la imagen que pueden ser reconocidos en diferentes fotogramas. Detectores comunes son ORB (Implementado en este proyecto), SIFT y AKAZE. Cada feature incluye un descriptor que permite comparar puntos entre imágenes.

En la etapa de matching, los descriptores se comparan entre dos frames consecutivos para encontrar correspondencias. Este proceso utiliza métricas como distancia Hamming (para ORB) o métodos aproximados como FLANN. Para eliminar coincidencias incorrectas se utilizan técnicas como ratio test y RANSAC.

Una vez que se tienen correspondencias confiables, se estima la matriz esencial, que describe la relación geométrica entre dos cámaras. Esto se calcula con cv2.findEssentialMat(). Posteriormente se recupera la pose relativa entre las cámaras usando cv2.recoverPose(), obteniendo una rotación (R) y una traslación (t).

Con estas poses se realiza la triangulación, que consiste en calcular la posición tridimensional de los puntos observados en dos imágenes distintas. Esto produce una nube de puntos 3D, que constituye el mapa inicial del entorno.

Finalmente, el sistema acumula las poses estimadas para reconstruir la trayectoria de la cámara.
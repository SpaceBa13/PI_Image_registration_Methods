import cv2

import cv2
import numpy as np

class ORBFeatureExtractor:
    def __init__(self, n_features=2500):
        # Aumentamos ligeramente nfeatures porque el filtrado posterior eliminará algunos
        self.orb = cv2.ORB_create(
            nfeatures=n_features,
            scaleFactor=1.1,      # Escala más fina (1.1 en vez de 1.2) para mejor precisión
            nlevels=12,           # Más niveles para manejar mejor el zoom del S24
            fastThreshold=20      # Umbral para detectar puntos en texturas finas
        )

    def extract(self, image):
        # 1. Pre-procesamiento
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # 2. Detección base (Coordenadas enteras)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        if not keypoints:
            return [], None

        # ---------------------------------------------------------
        # 3. REFINAMIENTO SUB-PÍXEL (El cambio clave)
        # ---------------------------------------------------------
        # Extraemos las coordenadas (u, v) de los keypoints
        pts = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        
        # Criterios de parada: 40 iteraciones o un cambio menor a 0.001 px
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
        
        # Refinamos la posición buscando el centro de masa de la intensidad local
        # Una ventana de (5,5) es ideal para no alejarse del punto original
        pts_refined = cv2.cornerSubPix(gray, pts, (5, 5), (-1, -1), criteria)
        
        # Inyectamos las nuevas coordenadas en los objetos KeyPoint originales
        for i, kp in enumerate(keypoints):
            kp.pt = tuple(pts_refined[i])
        # ---------------------------------------------------------
            
        return keypoints, descriptors
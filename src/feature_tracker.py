import cv2
import numpy as np

class FeatureTracker:
    def __init__(self):
        # Parámetros para Lucas-Kanade (KLT)
        # Ajustados para el sensor del S24 (Zoom 2x requiere ventanas más grandes)
        self.lk_params = dict(
            winSize=(21, 21),      # Ventana de búsqueda: 21x21 es robusta para zoom
            maxLevel=3,            # Niveles de pirámide para captar movimientos rápidos
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

    def track(self, prev_img, curr_img, prev_pts):
        """
        Realiza el seguimiento de puntos del frame anterior al actual.
        """
        # Asegurar escala de grises
        if len(prev_img.shape) == 3:
            prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
        else:
            prev_gray, curr_gray = prev_img, curr_img

        # KLT requiere los puntos en formato float32
        p0 = np.float32(prev_pts).reshape(-1, 1, 2)

        # Calcular Flujo Óptico Piramidal
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, p0, None, **self.lk_params
        )

        # Filtrar puntos donde el seguimiento fue exitoso (status == 1)
        # st.ravel()==1 filtra los puntos que se perdieron o salieron del frame
        success = st.ravel() == 1
        pts_old = p0[success].reshape(-1, 2)
        pts_new = p1[success].reshape(-1, 2)

        return pts_old, pts_new, success
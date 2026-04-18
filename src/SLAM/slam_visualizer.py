import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

class SLAMVisualizer3D:
    def __init__(self, data_provider, tracker, motion_engine, map_builder, extractor):
        self.data_provider = data_provider
        self.tracker = tracker
        self.motion_engine = motion_engine
        self.map_builder = map_builder
        self.extractor = extractor
        
        # 1. Preparar Frame de Referencia (Origen)
        self.img1 = data_provider.load_frame(r"resources\Triangulation_test_videos\video_frames\frame_0000.jpg")
        kp1, _ = self.extractor.extract(self.img1)
        self.pts1_init = np.array([kp.pt for kp in kp1], dtype=np.float32)

        # 2. Configuración de la Figura
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(bottom=0.2)

        # 3. Widget de Control (Slider para saltar entre frames)
        ax_slider = plt.axes([0.25, 0.05, 0.5, 0.03])
        self.slider = Slider(ax_slider, 'Comparar con Frame #', 1, 100, valinit=10, valstep=1)
        self.slider.on_changed(self.update)

        # Historial para dibujar la trayectoria
        self.path_x, self.path_y, self.path_z = [0], [0], [0]

        self.update(10) # Render inicial
        plt.show()

    def update(self, val):
        frame_idx = int(val)
        frame_path = fr"resources\Triangulation_test_videos\video_frames\frame_{frame_idx:04d}.jpg"
        img2 = self.data_provider.load_frame(frame_path)
        
        if img2 is None: return

        # --- Pipeline de Procesamiento ---
        # A. Tracking de puntos desde el origen al frame seleccionado
        p1_t, p2_t, st = self.tracker.track(self.img1, img2, self.pts1_init)
        
        # B. Estimación de Movimiento
        R, t, mask = self.motion_engine.recover_camera_motion(p1_t, p2_t)
        
        if R is not None:
            mask_bool = mask.ravel() == 1
            # C. Triangulación
            p3d = self.map_builder.triangulate(R, t, p1_t[mask_bool], p2_t[mask_bool])
            
            # D. Cálculo de la posición de la cámara en el Mundo (C = -R^T * t)
            # Esto reconstruye el "camino" o trayectoria
            cam_pos = -R.T @ t
            
            # --- Renderizado ---
            self.ax.cla()
            
            # Dibujar Nube de Puntos (Filtrar por profundidad lógica)
            z_mask = (p3d[:, 2] > 0.1) & (p3d[:, 2] < 15)
            self.ax.scatter(p3d[z_mask, 0], p3d[z_mask, 1], p3d[z_mask, 2], 
                            c=p3d[z_mask, 2], cmap='magma', s=2, alpha=0.5)

            # Dibujar Cámara Actual (Samsung S24)
            self.ax.scatter(cam_pos[0], cam_pos[1], cam_pos[2], color='red', s=50, label='Cámara Actual')
            
            # Dibujar Origen (Donde empezó el video)
            self.ax.scatter(0, 0, 0, color='blue', s=50, label='Inicio (Frame 0)')

            # Configuración de Ejes Fijos (Crítico para ver la escala del camino)
            self.ax.set_xlim(-2, 2)
            self.ax.set_ylim(-2, 2)
            self.ax.set_zlim(0, 10) # El microchip suele estar a esta distancia
            
            self.ax.set_title(f"Trayectoria: Frame 0 vs Frame {frame_idx:04d}")
            self.ax.set_xlabel('X (Lateral)')
            self.ax.set_ylabel('Y (Vertical)')
            self.ax.set_zlabel('Z (Avance)')
            self.ax.invert_yaxis() # Alinear con vista humana
            
            self.fig.canvas.draw_idle()
# =========================================================
# PROYECTO: Geometric SLAM - Phase 1: Sparse Reconstruction
# Brayan Alpízar Elizondo
# Ingenieria en Computadores
# Tecnológico de Costa Rica (TEC)
# Proyecto de Investigación
# DISPOSITIVO: Samsung S24 (Sensor Calibrado Zoom 2x)
# =========================================================

import cv2
import numpy as np
from src.data_acquisition_provider import DataProvider
from src.feature_extraction_orb import ORBFeatureExtractor
from src.feature_matching_ransac import FeatureMatcher
from src.motion_estimation_geometry import MotionEstimator
from src.map_triangulation_3d import MapTriangulator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.feature_tracker import FeatureTracker


# -------------------------
# Main Geometric SLAM Pipeline
# -------------------------
# Inputs:
#   img1, img2 : Frame pair captured by the Samsung S24.
#   K, dist    : Intrinsic matrix and distortion coefficients.
#
# Description:
#   Orchestrates Phase 1 of the SLAM system. Executes ORB 
#   feature extraction, performs robust RANSAC matching, 
#   estimates relative camera motion, and generates a 
#   sparse 3D reconstruction via triangulation.
#
# Outputs:
#   Synchronized 2D match visualization and 3D point 
#   cloud inspection.
# -------------------------
def main():
    """
    Main execution block for the Computer Vision pipeline:
    1. Load intrinsic parameters (Calibration).
    2. Robust feature extraction (ORB).
    3. Essential matrix and pose estimation (RANSAC).
    4. 3D point triangulation and interactive visualization.
    """
    
    # -------------------------
    # 1. Initialization & Calibration
    # -------------------------
    # Description:
    #   Load camera parameters (K matrix and distortion) 
    #   previously calculated for the device's 2x lens.
    data_provider = DataProvider(calibration_path="calibration_results/")
    
    try:
        K, dist = data_provider.load_camera_parameters()
        print("[SYSTEM] Calibration parameters loaded successfully.")
    except FileNotFoundError as e:
        print(f"[ERROR] Calibration file not found: {e}")
        return

    # Initialize core pipeline modules
    tracker = FeatureTracker()
    extractor = ORBFeatureExtractor(n_features=2000)
    matcher = FeatureMatcher(K)
    motion_engine = MotionEstimator(K)
    map_builder = MapTriangulator(K)

    # -------------------------
    # 2. Data Acquisition
    # -------------------------
    # Description:
    #   Select frames with sufficient lateral baseline to 
    #   ensure triangulation with low geometric error.

    img1 = data_provider.load_frame(r"resources\Triangulation_test_videos\video_frames\frame_0000.jpg")
    img2 = data_provider.load_frame(r"resources\Triangulation_test_videos\video_frames\frame_0001.jpg")
    if img1 is None or img2 is None:
        print("[ERROR] Image loading failed.")
        return

    # 3. Pipeline con Tracking
    
    # A. Detección inicial solo en el Frame 1
    kp1, des1 = extractor.extract(img1)
    # Convertimos los keypoints a formato numpy (u, v) para el tracker
    pts1_input = np.array([kp.pt for kp in kp1], dtype=np.float32)

    # B. Tracking hacia el Frame 2 (reemplaza a FeatureMatcher)
    # El tracker nos dice dónde terminaron los puntos de img1 en img2
    pts1_tracked, pts2_tracked, status = tracker.track(img1, img2, pts1_input)
    
    print(f"[INFO] Puntos seguidos con éxito: {len(pts2_tracked)}")

    # C. Estimación de Movimiento (Usa los puntos del tracker)
    # RANSAC dentro de recover_camera_motion limpiará los errores del tracking
    R, t, pose_mask = motion_engine.recover_camera_motion(pts1_tracked, pts2_tracked)

    # D. Triangulación
    # Importante: Solo triangulamos los puntos que pasaron el filtro de RANSAC (pose_mask)
    pts1_final = pts1_tracked[pose_mask.ravel() == 1]
    pts2_final = pts2_tracked[pose_mask.ravel() == 1]
    
    points_3d = map_builder.triangulate(R, t, pts1_final, pts2_final)
    
    # 4. Visualización
    # visualize_slam_matches(img1, img2, pts1_final, pts2_final, points_3d)

    # Realizar el seguimiento
    pts1_tracked, pts2_tracked, status = tracker.track(img1, img2, pts1_input)

    # Generar la visualización tipo "trail"
    tracking_vis = draw_feature_tracking(img2, pts1_tracked, pts2_tracked)

    # Mostrar el resultado
    cv2.imshow("Feature Tracking Output", tracking_vis)
    cv2.waitKey(0)


    """

    # -------------------------
    # 3. Processing Pipeline
    # -------------------------
    
    # A. Feature Extraction (ORB)
    # Detect keypoints and generate binary descriptors.
    kp1, des1 = extractor.extract(img1)
    kp2, des2 = extractor.extract(img2)
    print(f"[INFO] Features detected: Frame1={len(kp1)}, Frame2={len(kp2)}")

    # B. Matching and RANSAC Filtering
    # Filter false correspondences using the epipolar constraint.
    pts1, pts2, match_mask = matcher.match(kp1, des1, kp2, des2)
    print(f"[INFO] Valid matches (Inliers): {len(pts1)}")

    # C. Motion Estimation
    # Recover camera Rotation (R) and Translation (t).
    R, t, pose_mask = motion_engine.recover_camera_motion(pts1, pts2)
    
    print("\n--- Pose Estimation (Camera Motion) ---")
    print(f"Rotation Matrix (R):\n{R}")
    print(f"Translation Vector (t):\n{t}")

    # D. 3D Map Generation (Triangulation)
    # Project 2D points into 3D space relative to the camera.
    points_3d = map_builder.triangulate(R, t, pts1, pts2)
    
    print(f"\n[INFO] Triangulated 3D points: {len(points_3d)}")
    print(f"Sample coordinates (X, Y, Z):\n{points_3d[:5]}")

    # -------------------------
    # 4. Visualization & Inspection
    # -------------------------
    # Description:
    #   Launch the synchronized dual-system:
    #   - OpenCV: 2D match and optical flow inspection.
    #   - Matplotlib: Interactive 3D point cloud for metrology.
    print(f"\n[SYSTEM] Starting synchronized visualization...")
    visualize_slam_matches(img1, img2, pts1, pts2, points_3d)
    
    # Block thread to keep windows open
    cv2.waitKey(0)
    """




def draw_feature_tracking(image, pts_old, pts_new):
    """
    Dibuja líneas de flujo (rojas) y puntos actuales (verdes) 
    para visualizar el movimiento de las características.
    """
    # Creamos una copia para no sobreescribir la imagen original
    vis_img = image.copy()
    
    # Si la imagen es en escala de grises, la convertimos a BGR para dibujar en color
    if len(vis_img.shape) == 2:
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)

    for i, (new, old) in enumerate(zip(pts_new, pts_old)):
        # Coordenadas de los puntos (deben ser enteros para OpenCV)
        a, b = new.ravel().astype(int)
        c, d = old.ravel().astype(int)

        # Dibujar la línea roja que representa la trayectoria (estela)
        vis_img = cv2.line(vis_img, (a, b), (c, d), (0, 0, 255), 2)
        
        # Dibujar el punto verde en la posición actual
        vis_img = cv2.circle(vis_img, (a, b), 4, (0, 255, 0), -1)

    return vis_img


# -------------------------
# SLAM Cloud Visualization
# -------------------------
# Description:
#   Generates an interactive 3D scatter plot of the triangulated 
#   points. Uses a picker event to link 3D points back to their 
#   original 2D match IDs.

def plot_slam_cloud(points_3d):
    # -------------------------
    # Initialization & Filtering
    # -------------------------
    # Description:
    #   Prepares the data by filtering outliers and setting up 
    #   the coordinate system (Y-Up for human-centric view).
    
    indices = np.arange(len(points_3d))
    # Filter points based on depth (Z) to remove noise
    mask = (points_3d[:, 2] > 0.1) & (points_3d[:, 2] < 20)
    
    p3d = points_3d[mask]
    original_ids = indices[mask]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # -------------------------
    # 3D Rendering
    # -------------------------
    # Inputs:
    #   p3d: Filtered 3D coordinates.
    #
    # Description:
    #   Draws the camera origin, the optical axis, and the 
    #   point cloud with a depth-based colormap.
    
    # Camera Origin (Samsung S24 Center)
    ax.scatter(0, 0, 0, color='red', s=100, label='Cámara (0,0,0)', zorder=10)
    ax.quiver(0, 0, 0, 0, 0, 1.5, color='red', arrow_length_ratio=0.1, label='Eje Óptico')

    # Point Cloud with Picker interaction
    X, Y, Z = p3d[:, 0], p3d[:, 1], p3d[:, 2]
    sc = ax.scatter(X, Y, Z, c=Z, cmap='magma', s=20, alpha=0.8, picker=True)

    # -------------------------
    # Interaction Logic
    # -------------------------
    def on_pick(event):
        idx_in_filtered = event.ind[0]
        match_id = original_ids[idx_in_filtered]
        point = p3d[idx_in_filtered]
        
        print(f"\n[INSPECCIÓN] ID Match 2D: {match_id}")
        print(f"Coordenadas: X={point[0]:.3f}, Y={point[1]:.3f}, Z={point[2]:.3f}")

    fig.canvas.mpl_connect('pick_event', on_pick)

    # -------------------------
    # Coordinate System Setup
    # -------------------------
    ax.set_xlabel('X (Derecha)')
    ax.set_ylabel('Y (Invertido: Arriba)')
    ax.set_zlabel('Z (Profundidad)')
    
    # Invert Y to align with World Coordinates (Human Perspective)
    ax.invert_yaxis() 
    ax.set_box_aspect([np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim()), np.ptp(ax.get_zlim())])

    ax.view_init(elev=25, azim=-90)
    plt.colorbar(sc, label='Distancia (Z)')
    plt.title("Inspección 3D SLAM - Clic para ver ID")
    plt.legend()
    
    print("[SISTEMA] Ventana 3D lista para inspección.")
    plt.show()


# -------------------------
# SLAM Interactive Match Inspector
# -------------------------
# Description:
#   Synchronizes the 2D feature matching view with the 3D map. 
#   Allows frame-by-frame match verification using a trackbar.

def visualize_slam_matches(img1, img2, pts1, pts2, points_3d):
    # -------------------------
    # Initialization
    # -------------------------
    # Description:
    #   Sets up the interactive mode for Matplotlib and creates 
    #   the side-by-side canvas for OpenCV.
    
    plt.ion() 
    plot_slam_cloud(points_3d) 

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    canvas_orig = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas_orig[:h1, :w1] = img1
    canvas_orig[:h2, w1:w1+w2] = img2

    # Window scaling for high-res S24 images
    screen_res = (960, 540)
    scale = min(screen_res[0] / canvas_orig.shape[1], screen_res[1] / canvas_orig.shape[0])
    
    win_name = "SLAM Interactive Inspection"
    cv2.namedWindow(win_name)

    def nothing(x): pass
    cv2.createTrackbar("Match ID", win_name, 0, len(pts1) - 1, nothing)

    # -------------------------
    # Main Visualization Loop
    # -------------------------
    # Description:
    #   Real-time rendering of matches. Syncs UI events between 
    #   OpenCV (Trackbar) and Matplotlib (3D Plot).

    while True:
        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        trackbar_id = cv2.getTrackbarPos("Match ID", win_name)
        temp_canvas = canvas_orig.copy()
        
        # 1. Draw Background Optical Flow (Sparse)
        for i in range(0, len(pts1), 25):
            p1, p2 = pts1[i], pts2[i]
            cv2.line(temp_canvas, (int(p1[0]), int(p1[1])), 
                     (int(p2[0] + w1), int(p2[1])), (0, 60, 0), 1)

        # 2. Highlight Selected Match
        p1_s, p2_s = pts1[trackbar_id], pts2[trackbar_id]
        c1, c2 = (int(p1_s[0]), int(p1_s[1])), (int(p2_s[0] + w1), int(p2_s[1]))
        
        cv2.line(temp_canvas, c1, c2, (0, 255, 0), 2)
        cv2.circle(temp_canvas, c1, 8, (255, 0, 0), -1)
        cv2.circle(temp_canvas, c2, 8, (255, 0, 0), -1)
        
        cv2.putText(temp_canvas, f"ID SELECCIONADO: {trackbar_id}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 3. Scale and Display
        res_w, res_h = int(temp_canvas.shape[1] * scale), int(temp_canvas.shape[0] * scale)
        cv2.imshow(win_name, cv2.resize(temp_canvas, (res_w, res_h)))

        # Update Matplotlib events
        plt.pause(0.01) 

        if cv2.waitKey(20) & 0xFF == 27: # ESC to exit
            break

    # -------------------------
    # Cleanup
    # -------------------------
    plt.ioff()
    cv2.destroyAllWindows()
    plt.close('all')


if __name__ == "__main__":
    main()
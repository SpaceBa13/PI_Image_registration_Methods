import cv2
import os

def extract_frames(video_path, output_folder, target_fps):
    """
    video_path: Ruta del video original.
    output_folder: Carpeta donde se guardarán los frames.
    target_fps: Cuántos frames por segundo queremos extraer.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"[SISTEMA] Carpeta creada: {output_folder}")

    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir el video.")
        return

    # Obtener el framerate original del video
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calcular cada cuántos cuadros hay que guardar uno (el "paso")
    # Si original_fps=60 y target_fps=2, hop = 30
    if target_fps > original_fps:
        print(f"[ADVERTENCIA] El target_fps ({target_fps}) es mayor al original ({original_fps}). Se guardarán todos.")
        hop = 1
    else:
        hop = round(original_fps / target_fps)

    print(f"[INFO] Video Original: {original_fps:.2f} FPS")
    print(f"[INFO] Extrayendo a: {target_fps} FPS (Guardando 1 de cada {hop} cuadros)")

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        
        if not ret:
            break

        # Solo guardamos si el cuadro actual es múltiplo del salto (hop)
        if frame_count % hop == 0:
            frame_name = f"frame_{saved_count:04d}.jpg"
            file_path = os.path.join(output_folder, frame_name)
            
            cv2.imwrite(file_path, frame)
            saved_count += 1
            
            # Imprimir progreso cada vez que guardamos un cuadro
            print(f"[PROCESO] Guardado: {frame_name} (Frame original index: {frame_count})")
            
        frame_count += 1

    cap.release()
    print(f"\n[EXITO] Proceso terminado.")
    print(f"Total frames leídos: {frame_count}")
    print(f"Total frames guardados: {saved_count}")

# --- CONFIGURACIÓN ---
mi_video = r"resources\Triangulation_test_videos\video_base.mp4"
mi_destino = r"resources\Triangulation_test_videos\video_frames"

# Cambia este valor al framerate que necesites para tu SLAM
target_fps = 4
extract_frames(mi_video, mi_destino, target_fps)
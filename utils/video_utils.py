import cv2

def read_video(video_path):
    # Crea un objeto VideoCapture para leer el video
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        # Lee un frame del video
        ret, frame = cap.read()
        # Si no se pudo leer el frame, sale del bucle
        if not ret:
            break
        frames.append(frame)
    return frames
    
def save_video(output_video_frames, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para el video
    out = cv2.VideoWriter(output_video_path, fourcc, 30, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)  # Escribe el frame en el archivo de salida
    out.release()  # Libera el objeto VideoWriter


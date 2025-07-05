# -*- coding: utf-8 -*-

# Importar las clases y funciones necesarias de nuestros módulos y librerías externas
from utils import read_video, save_video
from trackers import Tracker               # Nuestra clase para detección y tracking
from team_assigner import TeamAssigner     # Nuestra clase para asignar equipos
import cv2                                 # Librería OpenCV para manipulación de imágenes y videos

def main():
    # ------------------- 1. Lectura del Video de Entrada -------------------
    # Lee los frames del archivo de video especificado.
    # 'read_video' debería devolver una lista de frames (arrays NumPy).
    print("Leyendo video de entrada...")
    video_frames = read_video('input_videos/corto_futbol.mp4')
    print(f"Video leído, {len(video_frames)} frames obtenidos.")

    # ------------------- 2. Inicialización del Tracker -------------------
    # Crea una instancia de la clase Tracker.
    # Se le pasa la ruta al archivo del modelo YOLOv8 entrenado ('best.pt').
    # Este modelo es el resultado del Punto 4 de la entrega (entrenamiento).
    print("Inicializando el Tracker con el modelo 'best.pt'...")
    tracker = Tracker('best.pt')

    # ------------------- 3. Obtención de Tracks (Detección y Tracking) -------------------
    # Ejecuta la detección de objetos (con 'best.pt') y el tracking (con ByteTrack) en los frames.
    # IMPORTANTE: 'read_from_stub=False' fuerza la ejecución del modelo y el tracker.
    #             Esto es necesario para cumplir el Punto 5 de la entrega ("probarlo en un vídeo").
    #             Si pones 'read_from_stub=True', cargará resultados guardados previamente en 'stub_path'
    #             (útil para depurar o ejecutar más rápido después de la primera vez).
    # 'stub_path' es el archivo donde se guardarán (si read_from_stub=False) o cargarán (si True) los tracks.
    stub_file = 'stubs/track_stubs_futbol.pkl'
    print(f"Obteniendo tracks de objetos... (read_from_stub=False, se ejecutará el modelo)")
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=False, # ¡Poner en False para la entrega!
                                       stub_path=stub_file)
    print(f"Tracks obtenidos. Resultados guardados/cargados desde '{stub_file}'.")
    # 'tracks' debería ser un diccionario como:
    # {
    #    "players": [ {track_id: {"bbox": [...]}}, {track_id: ...}, ... ], -> Lista por frame
    #    "goalkeepers": [ {track_id: {"bbox": [...]}}, ... ],             -> Lista por frame
    #    "referees": [ {track_id: {"bbox": [...]}}, ... ],                -> Lista por frame
    #    "ball": [ {1: {"bbox": [...]}}, ... ],                           -> Lista por frame (ID fijo '1')
    #    "managers": [ {track_id: {"bbox": [...]}}, ... ]                 -> Lista por frame
    # }

    # === BLOQUE COMENTADO PARA ANÁLISIS INICIAL (OPCIONAL) ===
    # Este bloque se usó originalmente (ejecutándolo una sola vez con el break)
    # para extraer y guardar la imagen de un jugador del primer frame.
    # Útil para analizar visualmente los colores de la camiseta o para depurar
    # la lógica de asignación de colores en TeamAssigner.
    # Mantener comentado para la ejecución normal del pipeline.
    # # ------------------- (Opcional) Guardar imagen de un jugador -------------------
    # # Itera sobre los jugadores detectados en el primer frame
    # if tracks.get("players") and len(tracks["players"]) > 0 and tracks["players"][0]:
    #     for track_id, player in tracks["players"][0].items():
    #         # Obtiene las coordenadas del cuadro delimitador (bounding box)
    #         bbox = player['bbox']
    #         # Obtiene el primer frame del video (asegúrate de que exista)
    #         if len(video_frames) > 0:
    #             frame = video_frames[0]
    #             # Recorta la imagen del jugador usando las coordenadas del bounding box
    #             # Asegurarse de que las coordenadas son enteros y están dentro de los límites del frame
    #             y1, y2 = int(max(0, bbox[1])), int(min(frame.shape[0], bbox[3]))
    #             x1, x2 = int(max(0, bbox[0])), int(min(frame.shape[1], bbox[2]))
    #             if y1 < y2 and x1 < x2: # Asegurarse de que el recorte es válido
    #                 cropped_image = frame[y1:y2, x1:x2]
    #                 # Guarda la imagen recortada en un archivo
    #                 output_image_path = f'output_images/player_{track_id}_frame0.jpg'
    #                 # Crear directorio si no existe
    #                 import os
    #                 os.makedirs('output_images', exist_ok=True)
    #                 cv2.imwrite(output_image_path, cropped_image)
    #                 print(f"Imagen de jugador guardada en: {output_image_path}")
    #                 # Salir después de guardar la primera imagen encontrada
    #                 break
    #         break # Salir del bucle de jugadores después del primero
    # === FIN DEL BLOQUE COMENTADO ===

    # ------------------- 4. Interpolación de Posiciones del Balón -------------------
    # Si se detectó el balón, interpola sus posiciones para suavizar el movimiento
    # y rellenar frames donde pudo no ser detectado.
    if tracks.get("ball"): # Verifica si la key "ball" existe y tiene datos
         print("Interpolando posiciones del balón...")
         tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
         print("Interpolación del balón completada.")
    else:
         print("Advertencia: No se detectó el balón en ningún frame.")


    # ------------------- 5. Asignación de Equipos -------------------
    print("Iniciando asignación de equipos...")
    # Inicializar el asignador de equipos
    team_assigner = TeamAssigner()

    # -- 5.a. Determinar colores de equipo (basado en el primer frame) --
    # Combinar jugadores y porteros del primer frame para el clustering inicial.
    primer_frame_idx = 0 # Usaremos el primer frame (índice 0)
    if len(video_frames) > primer_frame_idx:
        # Obtener tracks del primer frame, verificando que existan y no estén vacíos
        tracks_players_frame0 = tracks.get("players", [])
        tracks_goalkeepers_frame0 = tracks.get("goalkeepers", [])

        initial_player_detections = tracks_players_frame0[primer_frame_idx] if len(tracks_players_frame0) > primer_frame_idx else {}
        initial_goalkeeper_detections = tracks_goalkeepers_frame0[primer_frame_idx] if len(tracks_goalkeepers_frame0) > primer_frame_idx else {}

        # Combinar los diccionarios de jugadores y porteros del primer frame
        initial_combined_detections = {**initial_player_detections, **initial_goalkeeper_detections}

        if initial_combined_detections:
            print(f"Asignando colores de equipo basados en el frame {primer_frame_idx}...")
            # Llama a la función que probablemente usa KMeans para agrupar colores
            team_assigner.assign_team_color(video_frames[primer_frame_idx],
                                            initial_combined_detections)
            print("Colores de equipo determinados.")
        else:
            # Advertencia si no hay jugadores/porteros en el primer frame para asignar colores
            print(f"Advertencia: No se detectaron jugadores ni porteros en el frame {primer_frame_idx}. No se pueden asignar colores iniciales.")
    else:
         print(f"Advertencia: El video no tiene suficientes frames ({len(video_frames)}) para usar el frame índice {primer_frame_idx}.")


    # -- 5.b. Asignar equipo a cada jugador/portero en cada frame --
    print("Asignando equipo a cada jugador/portero detectado en cada frame...")
    # Iterar sobre todos los frames procesados
    for frame_num in range(len(video_frames)):
        # Asegurarse de que el frame existe en los tracks (puede haber menos frames en tracks si hubo error)
        frame_exists_in_players = len(tracks.get("players", [])) > frame_num
        frame_exists_in_goalkeepers = len(tracks.get("goalkeepers", [])) > frame_num

        # Procesar jugadores si existen tracks para este frame
        if frame_exists_in_players:
            for item_id, track_data in tracks["players"][frame_num].items():
                # Obtener el equipo del jugador basado en su bbox en este frame
                team = team_assigner.get_player_team(video_frames[frame_num],
                                                     track_data['bbox'],
                                                     item_id)
                # Guardar la asignación de equipo y el color correspondiente en la estructura 'tracks'
                tracks["players"][frame_num][item_id]['team'] = team
                # Usar .get() para el color por si acaso (aunque 'team' debería estar en team_colors)
                tracks["players"][frame_num][item_id]['team_color'] = team_assigner.team_colors.get(team, (255, 255, 255)) # Blanco por defecto

        # Procesar porteros si existen tracks para este frame
        if frame_exists_in_goalkeepers:
             for item_id, track_data in tracks["goalkeepers"][frame_num].items():
                # Obtener el equipo del portero basado en su bbox en este frame
                team = team_assigner.get_player_team(video_frames[frame_num],
                                                     track_data['bbox'],
                                                     item_id)
                # Guardar la asignación de equipo y el color
                tracks["goalkeepers"][frame_num][item_id]['team'] = team
                tracks["goalkeepers"][frame_num][item_id]['team_color'] = team_assigner.team_colors.get(team, (255, 255, 255)) # Blanco por defecto
    print("Asignación de equipos completada para todos los frames.")


    # ------------------- 6. Dibujar Anotaciones -------------------
    # Llama al método del Tracker que dibuja las elipses, triángulos, IDs, etc.,
    # sobre los frames originales usando la información de 'tracks' (incluyendo colores de equipo).
    print("Dibujando anotaciones en los frames...")
    output_video_frames = tracker.draw_annotations(video_frames, tracks)
    print("Anotaciones dibujadas.")

    # ------------------- 7. Guardar Video de Salida -------------------
    # Guarda los frames anotados como un nuevo archivo de video.
    output_video_path = 'output_videos/corto_futbol_output_equipos.mp4'
    print(f"Guardando video de salida en '{output_video_path}'...")
    save_video(output_video_frames, output_video_path)
    print("Video de salida guardado.")
    print("¡Proceso completado!")


# Punto de entrada del script: si se ejecuta directamente, llama a la función main()
if __name__ == "__main__":
    main()
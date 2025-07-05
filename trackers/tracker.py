from ultralytics import YOLO
import supervision as sv
import pickle
import os
import cv2
import numpy as np
import pandas as pd
import sys
sys.path.append("/..")
from utils import get_center_of_bbox, get_bbox_width

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def interpolate_ball_positions(self, ball_positions):
        # Interpolación de posiciones del balón
        ball_positions = [x.get(1,{}).get("bbox",[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=["x1", "y1", "x2", "y2"])
        df_ball_positions = df_ball_positions.interpolate() # Interpolación de NaN
        df_ball_positions = df_ball_positions.bfill() # Rellena los valores NaN hacia atrás
        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions

    def detect_frames(self, frames):
        batch_size = 8
        detections = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            detections_batch = self.model.predict(frames[i:i + batch_size], conf=0.1)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            # Cargar los tracks desde el stub
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks = {
            "players": [],
            "referees": [],
            "ball": [],
            "managers": [],
            "goalkeepers": []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Tracking
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            tracks["managers"].append({})
            tracks["goalkeepers"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]
                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}
                if cls_id == cls_names_inv['manager']:
                    tracks["managers"][frame_num][track_id] = {"bbox":bbox}
                if cls_id == cls_names_inv['goalkeeper']:
                    tracks["goalkeepers"][frame_num][track_id] = {"bbox":bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}

        if stub_path is not None:
            # Save the tracks to a stub file
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)
        cv2.ellipse(frame, 
                    center = (x_center,y2), 
                    axes = (int(width), int(0.35*width)),
                    angle = 0.0,
                    startAngle = -45,
                    endAngle = 235,
                    color = color,
                    thickness = 2,
                    lineType = cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height = 20
        # Calcular las coordenadas del rectángulo
        # usando el centro de la elipse y el tamaño del rectángulo
        # y2 es la coordenada y de la parte inferior de la elipse
        x1_rectangle = (x_center - rectangle_width//2)
        x2_rectangle = (x_center + rectangle_width//2)
        y1_rectangle = (y2 - rectangle_height//2)
        y2_rectangle = (y2 + rectangle_height//2)

        if track_id is not None:
            cv2.rectangle(frame, 
                          (int(x1_rectangle), int(y1_rectangle)), 
                          (int(x2_rectangle), int(y2_rectangle)), 
                          color, 
                          cv2.FILLED)
            x1_text = x1_rectangle + 12
            y1_text = y1_rectangle + 15
            if track_id > 99:
                x1_text -= 10

            cv2.putText(frame,
                        f"{int(track_id)}",
                        (int(x1_text), int(y1_text)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0,0,0),
                        2            
            )
        return frame
    
    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x,_ = get_center_of_bbox(bbox)
        triangle_points = np.array([
            [x, y], 
            [x - 10, y - 20], 
            [x + 10, y - 20]
            ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED) # Lleno el triángulo
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 2) # Dibujo el contorno del triángulo
        return frame

    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            manager_dict = tracks["managers"][frame_num]

            # Dibuja los jugadores
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))  # Rojo por defecto
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

            # Dibuja los árbitros
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255)) # Amarillo

            # Dibuja el tgriangulo que identifica el balón
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0)) # Verde

            # Dibuja los managers
            for _, manager in manager_dict.items():
                frame = self.draw_ellipse(frame, manager["bbox"], (0, 0, 0)) # Negro

            # Dibuja los porteros
            for _, goalkeeper in tracks["goalkeepers"][frame_num].items():
                frame = self.draw_ellipse(frame, goalkeeper["bbox"], (255, 0, 255)) # Azul

            output_video_frames.append(frame)

        return output_video_frames


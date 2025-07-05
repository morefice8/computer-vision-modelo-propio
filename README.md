# Proyecto Detección y Seguimiento Deportivo (YOLOv8 + Supervision)

## Descripción

Este proyecto implementa un pipeline de visión por computadora para detectar y seguir (tracking) objetos en videos deportivos, específicamente partidos de fútbol. Utiliza un modelo YOLOv8 entrenado a medida para identificar jugadores, porteros, árbitros y el balón. Además, aplica el tracker ByteTrack de la librería `supervision` para asignar IDs consistentes a los objetos detectados a lo largo de los frames y una lógica de asignación de equipos basada en el color de las camisetas utilizando KMeans.

---

## Conjunto de Datos (Dataset - Roboflow)

El modelo YOLOv8 utilizado en este proyecto fue entrenado con un dataset personalizado creado en Roboflow.

*   **Plataforma:** Roboflow Universe
*   **Tipo de Proyecto:** Object Detection
*   **Etiquetas:**
    *   `player` (Jugador)
    *   `goalkeeper` (Portero)
    *   `referee` (Árbitro)
    *   `ball` (Balón )
    *   `manager` (Entrenador)
*   **Origen de Datos / Origine Dati:** Vídeo de un partido de fútbol `corto_futbol.mp4`.
*   **Imágenes Etiquetadas:** 46  imágenes etiquetadas.

*   **Enlace Público al Dataset:**
    *   **https://app.roboflow.com/my-first-project-9vcww/detecciondeportemaster/browse**

---

## Entrenamiento del Modelo

El entrenamiento se realizó siguiendo las pautas de la entrega:

1.  El dataset fue descargado de Roboflow.
2.  Se utilizó Google Colab para el entorno de entrenamiento (script `training\entrenamiento_modelo_propio_Michele_OREFICE.ipynb`).
3.  Se ajustó la estructura de carpetas del dataset descargado según los requerimientos de YOLOv8.
4.  Se entrenó un modelo partiendo de la arquitectura `yolov8s.pt`.
5.  El modelo resultante (`best.pt`) se descargó para ser utilizado en este script de inferencia.

---

## Funcionalidades Implementadas

*   **Carga de Modelo:** Carga el modelo YOLOv8 (`best.pt`) entrenado a medida.
*   **Detección de Objetos:** Detecta `player`, `goalkeeper`, `referee`, `ball` (y `manager` si aplica) en cada frame del video.
*   **Tracking de Objetos:** Utiliza `supervision.ByteTrack` para asignar y mantener IDs únicos a jugadores, porteros, árbitros y managers a través de los frames.
*   **Interpolación del Balón:** Interpola la posición del balón usando `pandas` para suavizar la trayectoria y rellenar frames donde no fue detectado.
*   **Asignación de Equipos:**
    *   Determina los colores principales de los equipos usando `sklearn.KMeans` en las detecciones del primer frame (jugadores + porteros).
    *   Asigna a cada jugador/portero detectado en cada frame a un equipo basado en el color predominante de su camiseta.
*   **Anotación de Video:** Dibuja elipses de colores (según el equipo) con IDs de tracking en jugadores/porteros/árbitros/managers y un triángulo en el balón sobre los frames del video.
*   **Salida de Video:** Guarda el video procesado con todas las anotaciones.
*   **Caching (Opcional):** Permite guardar (`stub_path`) y recargar (`read_from_stub=True`) los resultados del tracking para acelerar ejecuciones posteriores durante el desarrollo/depuración.

---

## Estructura de Archivos
.
├── input_videos/
│ └── corto_futbol.mp4 # Video de entrada (o el nombre que uses)
├── output_videos/
│ └── corto_futbol_output_equipos.mp4 # Video de salida generado
├── output_images/ # (Opcional) Imágenes recortadas guardadas
│ └── player_X_frame0.jpg
├── stubs/
│ └── track_stubs_futbol.pkl # Archivo caché de los tracks (generado)
├── trackers/
│ ├── init.py
│ └── tracker.py # Clase Tracker (Detección, Tracking, Dibujo)
├── team_assigner/
│ ├── init.py
│ └── team_assigner.py # Clase TeamAssigner (Asignación de equipos)
├── utils/
│ ├── init.py
│ ├── video_utils.py # Funciones de utilidad (read_video, save_video, etc.)
| └── bbox_utils.py  # Funciones de utilidad (get_center_of_bbox, get_bbox_width, etc.)
├── best.pt # El modelo YOLOv8 entrenado descargado
├── main.py # Script principal para ejecutar el pipeline
├── requirements.txt # Dependencias del proyecto
└── README.md # Este archivo


---

## Instalación

1.  **Crear un Entorno Virtual (Recomendado):**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```
2.  **Instalar Dependencias:** Asegúrate de tener un archivo `requirements.txt` con todas las librerías necesarias. Puedes generarlo en tu entorno activo con `pip freeze > requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```
    Las dependencias clave incluyen / Le dipendenze chiave includono: `ultralytics`, `supervision`, `opencv-python`, `scikit-learn`, `pandas`, `numpy`.
3.  **Colocar Archivos:**
    *   Asegúrate de que el modelo entrenado `best.pt` esté en la carpeta raíz del proyecto (o donde lo espere la clase `Tracker`).
    *   Coloca el video que quieres procesar dentro de la carpeta `input_videos/`.

---

## Uso

1.  **Activar el Entorno Virtual:**
    ```bash
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```
2.  **Ejecutar el Script Principal:**
    ```bash
    python main.py
    ```
3.  **Configuración `read_from_stub`:**
    *   En `main.py`, la variable `read_from_stub` dentro de la llamada a `tracker.get_object_tracks` controla si se ejecuta el modelo (`False`) o se cargan resultados previos (`True`). **Para la entrega o la primera ejecución, debe ser `False`.**
4.  **Verificar Salida:** El video procesado se guardará en la carpeta `output_videos/` con el nombre especificado en `main.py`.

---
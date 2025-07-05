# YOLOv8 Football Player and Ball Tracking

## ⚽ Description

This project provides a complete computer vision pipeline to detect and track objects in sports videos, with a focus on football matches. It leverages a custom-trained **YOLOv8** model to identify players, goalkeepers, referees, and the ball.

The pipeline integrates the **ByteTrack** tracker from the `supervision` library to assign and maintain consistent IDs for each detected object across frames. A key feature is the automatic team assignment, which uses a **KMeans clustering** algorithm to differentiate teams based on the dominant color of their jerseys.

---

## 📊 Dataset (Roboflow)

The YOLOv8 model was trained on a custom dataset created and hosted on Roboflow Universe.

*   **Project Type:** Object Detection
*   **Classes / Labels:**
    *   `player`
    *   `goalkeeper`
    *   `referee`
    *   `ball`
    *   `manager`
*   **Data Source:** Frames extracted from the `corto_futbol.mp4` video.
*   **Annotated Images:** 46
*   **Public Dataset Link:** [**Football-Detection-Master on Roboflow**](https://app.roboflow.com/my-first-project-9vcww/detecciondeportemaster/browse)

---

## 🧠 Model Training

The training process was conducted in a Google Colab environment (`training/entrenamiento_modelo_propio_Michele_OREFICE.ipynb`). A `yolov8s.pt` model was used as the base architecture and fine-tuned on the custom dataset. The resulting model with the best weights (`best.pt`) is used for inference in this project.

---

## ✨ Key Features

*   **Custom Model Loading:** Loads the custom-trained `best.pt` YOLOv8 model for inference.
*   **Multi-Class Object Detection:** Identifies players, goalkeepers, referees, and the ball in each frame.
*   **Object Tracking with ByteTrack:** Applies `supervision.ByteTrack` to assign a unique ID to each detected person, ensuring tracking consistency throughout the video.
*   **Ball Position Interpolation:** Fills in frames where the ball detection might fail by interpolating its position using `pandas`, resulting in a smoother trajectory.
*   **Automatic Team Assignment:**
    *   Analyzes detections in the first frame to determine the two main team colors using `sklearn.KMeans`.
    *   Assigns each tracked player to a team based on the dominant color of their jersey in subsequent frames.
*   **Rich Video Annotation:** Overlays colored ellipses (based on team assignment), tracking IDs, and a distinct marker for the ball on the output video.
*   **Results Caching:** Includes functionality to save (`stub_path`) and load (`read_from_stub=True`) tracking results, which significantly speeds up development and debugging by skipping the inference step on subsequent runs.

---

## 📁 File Structure
.
├── input_videos/
│ └── corto_futbol.mp4
├── output_videos/
│ └── corto_futbol_output_equipos.mp4
├── stubs/
│ └── track_stubs_futbol.pkl
├── trackers/
│ ├── init.py
│ └── tracker.py # Core Tracker class
├── team_assigner/
│ ├── init.py
│ └── team_assigner.py # TeamAssigner class
├── utils/
│ ├── init.py
│ ├── video_utils.py # Video I/O helpers
│ └── bbox_utils.py # Bounding box utilities
├── best.pt # Trained YOLOv8 model
├── main.py # Main script to run the pipeline
├── requirements.txt # Project dependencies
└── README.md # This file

---

## ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/morefice8/computer-vision-modelo-propio.git
    cd computer-vision-modelo-propio
    ```

2.  **Create and activate a virtual environment (Recommended):**
    ```bash
    # On Windows
    python -m venv venv
    .\venv\Scripts\activate

    # On macOS / Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    Key dependencies include: `ultralytics`, `supervision`, `opencv-python`, `scikit-learn`, `pandas`, and `numpy`.

4.  **Place necessary files:**
    *   Ensure the trained model `best.pt` is in the project's root directory.
    *   Place the input video you want to process inside the `input_videos/` folder.

---

## 🚀 Usage

1.  **Activate the virtual environment:**
    ```bash
    source venv/bin/activate
    ```

2.  **Run the main script:**
    ```bash
    python main.py
    ```

3.  **Configuration Note:**
    *   In `main.py`, the `read_from_stub` variable in the `tracker.get_object_tracks()` call controls the pipeline's behavior.
    *   Set to `False` for the first run or when processing a new video. This will run the full detection and tracking model.
    *   Set to `True` on subsequent runs to load cached tracking data from the `stubs/` directory, which is much faster.

4.  **Check the Output:** The processed video will be saved in the `output_videos/` directory.

---

## 🤔 Troubleshooting

*   **`ModuleNotFoundError: No module named 'some_library'`**:
    This error means a required Python library is not installed. Make sure you have activated your virtual environment (`source venv/bin/activate`) and run `pip install -r requirements.txt`.

*   **CUDA/GPU Issues**:
    If `ultralytics` fails to use the GPU, ensure you have a compatible version of CUDA and PyTorch installed. For CPU-only inference, no action is needed.

*   **File Not Found**:
    Ensure the input video is placed in the `input_videos/` folder and that the `best.pt` model file is in the root directory.

---

## 📜 License

This project is distributed under the MIT License. See the `LICENSE` file for more details.

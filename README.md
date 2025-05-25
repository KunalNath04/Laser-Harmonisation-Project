# Laser Harmonisation Project

This project implements a system for detecting a plus sign and a red dot in live video using Raspberry Pi and computer vision techniques, and integrates a capacitive sensing circuit for laser harmonisation feedback. The repository contains code for dataset generation, model training, object detection, red-dot localization, and Raspberry Pi GPIO interfacing.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Prerequisites](#prerequisites)
3. [Dataset Generation](#dataset-generation)

   * Synthetic Dataset Creation
   * Background Augmentation
4. [Model Training](#model-training)

   * YOLOv8 Plus-Sign Detector
   * PixelLib Instance Segmentation (optional)
5. [Object Detection & Red Dot Localization](#object-detection--red-dot-localization)

   * YOLOv8 Inference
   * Red Dot Detection (HSV Mask)
   * Distance & Direction Calculations
6. [Raspberry Pi Integration](#raspberry-pi-integration)

   * Picamera2 Configuration
   * GPIO Capacitive Sensor (harmonisation)
7. [File Structure](#file-structure)
8. [Usage](#usage)
9. [Troubleshooting](#troubleshooting)
10. [License](#license)

---

## Project Overview

The Laser Harmonisation Project combines computer vision and hardware interfacing to detect alignment of a red dot relative to a plus sign marker. A YOLOv8 model detects the plus sign in real-time video, while a simple HSV-based mask localizes the red dot. The system computes Euclidean distance and directional angle between the marker and the dot. An accompanying capacitive sensing circuit on a Raspberry Pi provides feedback on harmonisation status.

## Prerequisites

* Python 3.7+
* OpenCV
* NumPy
* Ultralytics YOLOv8 (`ultralytics` package)
* Picamera2 (for Raspberry Pi OS)
* RPi.GPIO (for Raspberry Pi GPIO)
* PixelLib & Mask R-CNN (optional for instance segmentation)

Install dependencies:

```bash
pip install opencv-python numpy ultralytics picamera2 RPi.GPIO pixellib
```

---

## Dataset Generation

### Synthetic Dataset Creation

The script in `dataset_generation/plus_sign_generator.py` produces `num_signs` synthetic images of plus signs on a Cartesian plane and generates corresponding XML annotations (Pascal VOC format).

* **Parameters**:

  * `plane_size`: Canvas size (pixels)
  * `num_signs`: Number of images to generate
  * `arm_length`: Half-length of plus sign arms
  * `line_width`: Plus sign thickness

### Background Augmentation

The script in `dataset_generation/background_overlay.py` overlays a transparent plus sign template onto a variety of background images at random scales and positions, saving results in `data/output`.

## Model Training

### YOLOv8 Plus-Sign Detector

Place your annotated dataset in YOLO format under `yolo_dataset/` and update the `data.yaml` file. Train using:

```bash
yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=300 imgsz=640
```

The resulting `.pt` model will be used for inference in `detect_plus_red.py`.

### PixelLib Instance Segmentation (Optional)

Use `pixellib_train.py` to train a Mask R-CNN model on a custom dataset:

```python
from pixellib.custom_train import instance_custom_training

# Example
dataset_path = "path/to/dataset"
num_classes = 2  # plus sign + background
model_name = "my_trained_model"
train_pixellib_model(dataset_path, num_classes, model_name)
```

Process video using:

```python
detect_objects_in_video("input.mp4", "my_trained_model/mask_rcnn_model.h5", "output.mp4")
```

---

## Object Detection & Red Dot Localization

Code file: `detect_plus_red.py`

1. **Load YOLO Model**

   ```python
   model = YOLO("/path/to/red_plus.pt")
   ```

2. **Frame Capture via Picamera2**

3. **Red Dot Detection** using HSV thresholding in `find_red_dot()`

4. **Process YOLO Detections**:

   * Draw bounding box around plus sign
   * Compute center coordinate
   * If red dot found, draw circle and compute:

     * Euclidean distance (`calculate_distance`)
     * Direction angle (`calculate_direction`)

5. **Display** original frame and zoomed crop with annotations

---

## Raspberry Pi Integration

### Picamera2 Configuration

In `detect_plus_red.py`, configure the camera:

```python
picam2 = Picamera2()
picam2.video_configuration.controls.FrameRate = 30
picam2.configure(picam2.create_preview_configuration(main={"format": 'RGB888', "size": (1280, 720)}))
picam2.start()
```

### GPIO Capacitive Sensor (Harmonisation)

Script: `rc_sensor.py`

* Measures charge time on pin 7
* Returns `"harmonisation is done"` when count ≤ 50

```python
def rc_time(pin):
    # discharge then measure
    ...
    return "harmonisation is done" if count<=50 else "In process"
```

---

## File Structure

```
├── dataset_generation/
│   ├── plus_sign_generator.py
│   └── background_overlay.py
├── yolo_dataset/
│   ├── images/
│   ├── labels/
│   └── data.yaml
├── detect_plus_red.py
├── pixellib_train.py
├── rc_sensor.py
├── requirements.txt
└── README.md
```

---

## Usage

1. Generate or prepare dataset
2. Train YOLO model or PixelLib model
3. Copy trained model to Raspberry Pi path
4. Run detection:

   ```bash
   python3 detect_plus_red.py
   ```
5. Monitor output windows; press `q` to quit
6. Run capacitive sensor script:

   ```bash
   python3 rc_sensor.py
   ```

---

## Troubleshooting

* **No red dot detected**: Adjust HSV ranges in `find_red_dot()`
* **Weak YOLO performance**: Increase dataset size or epochs
* **GPIO errors**: Ensure correct pin numbering mode and wiring



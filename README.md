**Project Overview**
The Laser Harmonisation Project combines computer vision and hardware interfacing to detect alignment of a red dot relative to a plus sign marker. A YOLOv8 model detects the plus sign in real-time video, while a simple HSV-based mask localizes the red dot. The system computes Euclidean distance and directional angle between the marker and the dot. An accompanying capacitive sensing circuit on a Raspberry Pi provides feedback on harmonisation status.

**Prerequisites**
Python 3.7+
OpenCV
NumPy
Ultralytics YOLOv8 (ultralytics package)
Picamera2 (for Raspberry Pi OS)
RPi.GPIO (for Raspberry Pi GPIO)
PixelLib & Mask R-CNN (optional for instance segmentation)

**Install dependencies:**
pip install opencv-python numpy ultralytics picamera2 RPi.GPIO pixellib
Dataset Generation
Synthetic Dataset Creation
The script in dataset_generation/plus_sign_generator.py produces num_signs synthetic images of plus signs on a Cartesian plane and generates corresponding XML annotations (Pascal VOC format).

**Parameters:**
plane_size: Canvas size (pixels)
num_signs: Number of images to generate
arm_length: Half-length of plus sign arms
line_width: Plus sign thickness
Background Augmentation
The script in dataset_generation/background_overlay.py overlays a transparent plus sign template onto a variety of background images at random scales and positions, saving results in data/output.
Model Training
YOLOv8 Plus-Sign Detector
Place your annotated dataset in YOLO format under yolo_dataset/ and update the data.yaml file. Train using: yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=300 imgsz=640
The resulting .pt model will be used for inference in detect_plus_red.py.

Object Detection & Red Dot Localization
Code file: detect_plus_red.py

**Load YOLO Model ** 
1) model = YOLO("/path/to/red_plus.pt")
2) Frame Capture via Picamera2
3)Red Dot Detection using HSV thresholding in find_red_dot()
4)Process YOLO Detections:
  Draw bounding box around plus sign
  Compute center coordinate
  If red dot found, draw circle and compute:
     Euclidean distance (calculate_distance)
     Direction angle (calculate_direction)
5)Display original frame and zoomed crop with annotations

**Raspberry Pi Integration**
Picamera2 Configuration
In detect_plus_red.py, configure the camera:
picam2 = Picamera2()
picam2.video_configuration.controls.FrameRate = 30
picam2.configure(picam2.create_preview_configuration(main={"format": 'RGB888', "size": (1280, 720)}))
picam2.start()

**GPIO Capacitive Sensor (Harmonisation)**
Script: rc_sensor.py
Measures charge time on pin 7
Returns "harmonisation is done" when count ≤ 50

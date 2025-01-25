import os
import cv2
import numpy as np
import tensorflow as tf

# Function to load YOLO model for plus sign detection
def load_yolo_model(model_path):
    yolo = tf.keras.models.load_model(model_path)
    return yolo

# Function to load custom CNN model for red dot detection
def load_red_dot_model(model_path):
    red_dot_model = tf.keras.models.load_model(model_path)
    return red_dot_model

# Function to load custom CNN model for plus sign center detection
def load_plus_center_model(model_path):
    plus_center_model = tf.keras.models.load_model(model_path)
    return plus_center_model

# Function to detect plus sign centers using CNN model
def detect_plus_sign_center(image, plus_center_model):
    # Preprocess the image for plus sign center detection
    processed_image = preprocess_image_for_plus_center(image)
    
    # Use plus center model to predict center coordinates
    center_coords = plus_center_model.predict(processed_image)
    
    return center_coords

# Function to preprocess image for plus sign center detection
def preprocess_image_for_plus_center(image):
    # Implement necessary preprocessing for plus center model input
    processed_image = image  # Placeholder for actual preprocessing
    return processed_image

# Function to extract image region around a specific plus sign center
def extract_plus_sign_region(image, center_coords, region_size=50):
    x_center, y_center = center_coords
    
    # Extract a square region around the center coordinates
    top_left_x = int(x_center - region_size / 2)
    top_left_y = int(y_center - region_size / 2)
    bottom_right_x = int(x_center + region_size / 2)
    bottom_right_y = int(y_center + region_size / 2)
    
    plus_sign_region = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    
    return plus_sign_region

# Function to detect red dot within a specific region using CNN model
def detect_red_dot(image_region, red_dot_model):
    # Preprocess the image region for red dot detection
    processed_region = preprocess_image_for_red_dot(image_region)
    
    # Use red dot model to predict red dot coordinates
    red_dot_coords = red_dot_model.predict(processed_region)
    
    return red_dot_coords

# Function to preprocess image for red dot detection
def preprocess_image_for_red_dot(image_region):
    # Implement necessary preprocessing for red dot model input
    processed_region = image_region  # Placeholder for actual preprocessing
    return processed_region

# Function to calculate distance and direction between plus sign center and red dot
def calculate_distance_and_direction(plus_center_coords, red_dot_coords):
    center_x, center_y = plus_center_coords
    red_dot_x, red_dot_y = red_dot_coords
    
    # Calculate distance
    distance = np.sqrt((red_dot_x - center_x)**2 + (red_dot_y - center_y)**2)
    
    # Calculate direction (angle)
    direction = np.arctan2(red_dot_y - center_y, red_dot_x - center_x) * (180 / np.pi)
    
    return distance, direction

# Define absolute paths to model files
yolo_model_path = os.path.abspath('/Users/dvirani/dp/best.pt')
red_dot_model_path = os.path.abspath('/Users/dvirani/dp/trails/points.h5')
plus_center_model_path = os.path.abspath('/Users/dvirani/dp/trails/plus.h5')

# Load your custom YOLO model for plus sign detection
yolo_model = load_yolo_model(yolo_model_path)

# Load your custom CNN model for red dot detection
red_dot_model = load_red_dot_model(red_dot_model_path)

# Load your custom CNN model for plus sign center detection
plus_center_model = load_plus_center_model(plus_center_model_path)

# Load and preprocess input image (e.g., from camera or file)
input_image_path = '/Users/dvirani/dp/trails/IMG_3922.jpg'
input_image = cv2.imread(input_image_path)

# Detect plus signs using YOLO model
plus_sign_boxes = detect_plus_signs(input_image, yolo_model)

# Assuming you identify the specific plus sign and its center coordinates
chosen_plus_center = [x_center, y_center]  # Get this from your plus sign detection results

# Detect the center of the chosen plus sign using the plus center model
plus_sign_center_coords = detect_plus_sign_center(input_image, plus_center_model)

# Extract the region around the detected plus sign center
plus_sign_region = extract_plus_sign_region(input_image, plus_sign_center_coords)

# Detect red dot within the extracted region using CNN model
red_dot_coords = detect_red_dot(plus_sign_region, red_dot_model)

# Calculate distance and direction between plus sign center and red dot
distance, direction = calculate_distance_and_direction(plus_sign_center_coords, red_dot_coords)

# Output the distance and direction
print(f"Distance from plus sign center: {distance} units")
print(f"Direction from plus sign center: {direction} degrees")

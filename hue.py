from ultralytics import YOLO
import cv2
import math
import numpy as np
from picamera2 import Picamera2

def find_red_dot(frame):
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds for the red color (you may need to adjust these)
    hue_range = 10  # Range of ±10 around the given hue value
    saturation_range = 70  # Range of ±50 around the given saturation value
    value_range = 70  # Range of ±50 around the given value value

    lower_red = np.array([177 - hue_range, 196 - saturation_range, 185 - value_range])
    upper_red = np.array([177 + hue_range, 196 + saturation_range, 185 + value_range])

    # Threshold the HSV image to get only red colors
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contour is found
    if contours:
        # Get the largest contour
        contour = max(contours, key=cv2.contourArea)

        # Calculate the centroid of the contour
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            return cx, cy

    return None

# Calculate Euclidean distance between two points
def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Calculate direction in degrees between two points
def calculate_direction(origin, point):
    x1, y1 = origin
    x2, y2 = point
    # Calculate relative coordinates from the origin
    rel_x = x2 - x1
    rel_y = y2 - y1
    # Calculate angle in radians
    angle_rad = math.atan2(rel_y, rel_x)
    # Convert angle to degrees
    angle_deg = math.degrees(angle_rad)
    # Ensure the angle is between 0 and 360 degrees
    if angle_deg < 0:
        angle_deg += 360
    return angle_deg

def main():
    # Initialize the PiCamera
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"format": 'RGB888', "size": (1280, 720)}))
    picam2.start()

    # Load your pre-trained model (ensure compatibility with YOLOv8)
    model = YOLO("/home/DP_kunal/DP_73/red_plus.pt")

    # Update class names (only plus sign)
    # Update class names (include all classes detected by the YOLO model)
    classNames = ["background", "plus_sign", "red_dot", ...]

    # Optional: Confidence threshold
    confidence_threshold = 0.5  # Adjust this value as needed (between 0 and 1)
    
    try:
        while True:
            # Capture frame from Raspberry Pi camera
            frame = picam2.capture_array()

            # Find red dot
            red_dot_coords = find_red_dot(frame)

            results = model(frame, stream=True)

            # Process detections
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Bounding Box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1

                    # Confidence
                    conf = math.ceil((box.conf[0] * 100)) / 100

                    # Display class name and confidence
                    if conf > confidence_threshold:
                        class_id = int(box.cls[0])  # Get the class ID
                        class_name = classNames[class_id]  # Get the class name

                        # We know the class is plus sign since we removed other classes

                        # Green for plus sign
                        color = (0, 255, 0)

                        # Draw bounding box and label
                        text = f'{class_name}: {conf}'
                        cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                        # Calculate center coordinates
                        center_x = int((x1 + x2) / 2)-470
                        center_y = -(int((y1 + y2) / 2))+305
                        center_text = f'Center: ({center_x}, {center_y})'
                        cv2.putText(frame, center_text, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        # Draw red dot if found and calculate distance and direction
                        if red_dot_coords:
                            cv2.circle(frame, red_dot_coords, 5, (0, 0, 255), -1)
                            cv2.putText(frame, f'Red Dot: {red_dot_coords}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                                        2)

                            # Calculate distance between plus sign center and red dot
                            distance = calculate_distance((center_x, center_y), red_dot_coords)
                            cv2.putText(frame, f'Distance: {distance:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                                        2)

                            # Calculate direction between plus sign center and red dot
                            direction = calculate_direction((center_x, center_y), red_dot_coords)
                            cv2.putText(frame, f'Direction: {direction:.2f} degrees', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (0, 0, 255), 2)
                        else:
                            cv2.putText(frame, 'No red dot', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Display image
            cv2.imshow('Webcam', frame)

            # Exit on 'q' press
            if cv2.waitKey(1) == ord('q'):
                break

    finally:
        # Release resources
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

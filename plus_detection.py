import cv2
import numpy as np

def find_laser_dot(frame):
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds for the laser color (you may need to adjust these)
    lower_laser = np.array([0, 100, 100])
    upper_laser = np.array([10, 255, 255])

    # Threshold the HSV image to get only laser colors
    mask = cv2.inRange(hsv, lower_laser, upper_laser)

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

def main():
    # Open a connection to the camera
    cap = cv2.VideoCapture(0)  # 0 for the default camera

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)

        # Find the laser dot
        laser_dot_coords = find_laser_dot(frame)

        # Calculate the center of the video frame
        center_x = frame.shape[1] // 2
        center_y = frame.shape[0] // 2

        if laser_dot_coords:
            # Calculate coordinates relative to the center of the video frame
            relative_x = laser_dot_coords[0] - center_x
            relative_y = laser_dot_coords[1] - center_y

            print("Coordinates of laser light relative to center (0,0):", (relative_x, relative_y))
        else:
            print("No laser light")

        # Display the frame
        cv2.imshow("Frame", frame)

        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


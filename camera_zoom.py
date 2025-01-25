import cv2
import numpy as np

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

def main():
    # Open a connection to the video file or camera
    cap = cv2.VideoCapture(0)  # 0 for the default camera

    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)

        # Find the red dot
        red_dot_coords = find_red_dot(frame)

        # Calculate the center of the video frame
        center_x = frame.shape[1] // 2
        center_y = frame.shape[0] // 2

        # Draw x and y axes
        cv2.line(frame, (0, center_y), (frame.shape[1], center_y), (0, 255, 0), 2)
        cv2.line(frame, (center_x, frame.shape[0]), (center_x, 0), (0, 255, 0), 2)  # Inverted y-axis

        if red_dot_coords:
            # Calculate coordinates relative to the center of the video frame
            relative_x = red_dot_coords[0] - center_x
            relative_y = center_y - red_dot_coords[1]  # Invert y-axis

            print("Coordinates of red dot relative to center (0,0):", (relative_x, relative_y))

            # Draw a circle at the red dot
            cv2.circle(frame, red_dot_coords, 5, (0, 0, 255), -1)
        else:
            print("No red dot found")

        # Display the frame
        cv2.imshow("Frame", frame)

        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Release the video file or camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

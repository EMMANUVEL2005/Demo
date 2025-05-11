import cv2
import numpy as np

def detect_traffic_lights(frame):
    """
    Detect red, yellow, and green traffic lights in the frame using color segmentation.
    Returns the frame with detected lights circled.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define color ranges for red, yellow, and green in HSV
    # Red has two ranges due to HSV wrap-around
    red_lower1 = np.array([0, 100, 100])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([160, 100, 100])
    red_upper2 = np.array([179, 255, 255])

    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])

    green_lower = np.array([40, 50, 50])
    green_upper = np.array([90, 255, 255])

    # Create masks for each color
    mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
    mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
    mask_green = cv2.inRange(hsv, green_lower, green_upper)

    # Detect contours for each mask and draw circles around detected lights
    for mask, color in [(mask_red, (0,0,255)), (mask_yellow, (0,255,255)), (mask_green, (0,255,0))]:
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 50:  # filter small contours
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                center = (int(x), int(y))
                radius = int(radius)
                if radius > 5:
                    cv2.circle(frame, center, radius, color, 2)
                    cv2.putText(frame, f"{color} light", (center[0]-10, center[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame

def main():
    cap = cv2.VideoCapture(0)  # capture from webcam

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output_frame = detect_traffic_lights(frame)

        cv2.imshow("Traffic Signal Detection", output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

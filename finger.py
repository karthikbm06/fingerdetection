import cv2
import numpy as np

# Function to count the number of fingers
def count_fingers(contour):
    if len(contour) < 5:
        return 0

    # Find the convex hull and ensure it's valid
    hull = cv2.convexHull(contour, returnPoints=False)
    
    if len(hull) < 3:
        return 0

    # Calculate convexity defects only if hull size is valid
    defects = cv2.convexityDefects(contour, hull)
    if defects is None:
        return 0

    finger_count = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])

        # Calculate the distances and angles between the points
        a = np.linalg.norm(np.array(end) - np.array(start))
        b = np.linalg.norm(np.array(far) - np.array(start))
        c = np.linalg.norm(np.array(end) - np.array(far))

        # Apply the cosine rule to find the angle in radians, then convert to degrees
        angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c))
        angle_deg = np.degrees(angle)

        # If the angle is less than 90 degrees and defect depth is significant, count as a finger
        if angle_deg <= 90 and d > 10000:
            finger_count += 1

    return finger_count

# Initialize video capture
cap = cv2.VideoCapture(0)

# Set video frame width and height
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Define the region of interest (ROI) for hand detection
    roi = frame[100:400, 200:500]

    # Convert the ROI to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)

    # Thresholding to create a binary image
    _, threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        # Find the largest contour, assuming it's the hand
        max_contour = max(contours, key=cv2.contourArea)

        # Approximate the contour to a simpler shape
        epsilon = 0.02 * cv2.arcLength(max_contour, True)
        approx = cv2.approxPolyDP(max_contour, epsilon, True)

        # Count the number of fingers
        finger_count = count_fingers(max_contour)

        # Draw the contours on the ROI
        cv2.drawContours(roi, [max_contour], -1, (0, 255, 0), 2)
        cv2.drawContours(roi, [cv2.convexHull(max_contour)], -1, (0, 0, 255), 2)

        # Display the finger count on the frame
        cv2.putText(frame, f"Fingers: {finger_count + 1}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the result
    cv2.imshow('Hand Region', roi)
    cv2.imshow('Detected Fingers in Video', frame)

    # Press 'q' to exit the video stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the windows
cap.release()
cv2.destroyAllWindows()

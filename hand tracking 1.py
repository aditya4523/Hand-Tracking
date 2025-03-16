import cv2
import numpy as np
import mediapipe as mp
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Open the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

# Capture initial frame to get dimensions
ret, frame = cap.read()
if not ret or frame is None:
    print("Error: Unable to read the initial frame.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

height, width = frame.shape[:2]
trail = np.zeros((height, width, 3), dtype=np.uint8)

# Store points with timestamps
point_buffer = []
TRAIL_DURATION = 10  # Duration to keep trails (seconds)

while True:
    # Capture the current frame
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error: Unable to read frame.")
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert frame to RGB (MediaPipe requirement)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    # Get the current time
    current_time = time.time()

    # Check if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the index finger tip position (landmark 8)
            index_x = int(hand_landmarks.landmark[8].x * width)
            index_y = int(hand_landmarks.landmark[8].y * height)

            # Add the point with timestamp to buffer
            point_buffer.append(((index_x, index_y), current_time))

            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Remove old points (older than TRAIL_DURATION seconds)
    point_buffer = [
        (point, timestamp) for point, timestamp in point_buffer
        if current_time - timestamp <= TRAIL_DURATION
    ]

    # Fade the trail slightly for a smooth effect
    trail = cv2.addWeighted(trail, 0.98, np.zeros_like(trail), 0.02, 0)

    # Draw the trail with dynamic color
    for i, (point, _) in enumerate(point_buffer):
        color = (0, 255 - (i % 255), 255)  # Vary color over time
        cv2.circle(trail, point, 5, color, -1)

    # Combine the trail with the current frame
    combined = cv2.addWeighted(frame, 0.7, trail, 0.3, 0)

    # Show the output
    cv2.imshow("Hand Movement Tracing (Index Finger)", combined)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

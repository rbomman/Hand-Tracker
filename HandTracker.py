import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    """Calculates the angle between three points using the law of cosines."""
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)

def create_info_image(finger_states, width=600, height=400, color=(255, 255, 255)):
    """Creates an image displaying the state of the index and middle fingers."""
    img = np.zeros((height, width, 3), np.uint8)
    img[:] = color

    for i, (finger, state) in enumerate(finger_states.items()):
        text = f"{finger}: {state}"
        cv2.putText(img, text, (10, 50 + i * 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    return img

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    finger_states = {"Thumb": "Unknown", "Index Finger": "Unknown", "Middle Finger": "Unknown", "Ring Finger": "Unknown", "Pinky Finger": "Unknown"}

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Drawing landmarks on the hand
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Calculate angles and determine state for index and middle fingers
            for finger_name, finger_indices in [("Thumb", [1, 2, 4]), ("Index Finger", [5, 6, 8]), ("Middle Finger", [9, 10, 12]), ("Ring Finger", [13, 14, 16]), ("Pinky Finger", [17, 19, 20])]:
                # Get landmark positions
                landmarks = [hand_landmarks.landmark[i] for i in finger_indices]
                # Convert positions to tuples (x, y) * frame width/height to scale to pixel space
                positions = [(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in landmarks]
                # Calculate the angle
                angle = calculate_angle(positions[0], positions[1], positions[2])
                # Determine the state based on the angle threshold
                finger_states[finger_name] = "Rested" if angle > 160 else "Activated"

    info_img = create_info_image(finger_states)

    cv2.imshow('Hand Tracking', frame)
    cv2.imshow('Finger States', info_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

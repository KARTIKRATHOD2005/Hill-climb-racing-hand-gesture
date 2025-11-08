import cv2
import mediapipe as mp
from pynput.keyboard import Controller, Key
from time import sleep

# Keyboard controller (simulating key presses)
keyboard = Controller()

# MediaPipe hand tracking solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)  # Increased reliability
mp_drawing = mp.solutions.drawing_utils

# Start the webcam
cap = cv2.VideoCapture(0)

# Variable initialization
is_gas_pressed = False
is_brake_pressed = False

# Hand fist (closed hand) detection function
def is_fist(landmarks):
    thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    little_tip = landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    
    if (index_tip.y > landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y and 
        middle_tip.y > landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y and 
        ring_tip.y > landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y and 
        little_tip.y > landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y):
        return True
    return False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the video for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for hand detection
    results = hands.process(rgb_frame)

    # If hands are detected
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Draw the hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            # Color the landmarks with different colors
            for idx, landmark in enumerate(landmarks.landmark):
                # Get coordinates
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                
                # Different colors for each finger's tip
                if idx == mp_hands.HandLandmark.THUMB_TIP:
                    color = (0, 0, 255)  # Red for Thumb
                elif idx == mp_hands.HandLandmark.INDEX_FINGER_TIP:
                    color = (0, 255, 0)  # Green for Index Finger
                elif idx == mp_hands.HandLandmark.MIDDLE_FINGER_TIP:
                    color = (255, 0, 0)  # Blue for Middle Finger
                elif idx == mp_hands.HandLandmark.RING_FINGER_TIP:
                    color = (0, 255, 255)  # Yellow for Ring Finger
                elif idx == mp_hands.HandLandmark.PINKY_TIP:
                    color = (255, 0, 255)  # Purple for Pinky Finger
                else:
                    color = (255, 255, 255)  # White for others

                # Draw a circle on each landmark
                cv2.circle(frame, (x, y), 5, color, -1)

            # Initialize status text
            status_text = ""

            # Check if fist gesture is detected
            if is_fist(landmarks):  
                if landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x > 0.5:  # Right hand (larger x value)
                    if not is_gas_pressed:
                        print("Right hand closed: Accelerator (Gas)")
                        keyboard.press(Key.right)
                        is_gas_pressed = True
                        is_brake_pressed = False
                    status_text = "Accelerator (Gas)"

                elif landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x < 0.5:  # Left hand (smaller x value)
                    if not is_brake_pressed:
                        print("Left hand closed: Brake")
                        keyboard.press(Key.left)
                        is_brake_pressed = True
                        is_gas_pressed = False
                    status_text = "Brake"

                sleep(0.1)

            else:
                if is_gas_pressed:
                    keyboard.release(Key.right)
                    is_gas_pressed = False
                if is_brake_pressed:
                    keyboard.release(Key.left)
                    is_brake_pressed = False

            # Display the status text on the frame if an action is detected
            if status_text:
                cv2.putText(frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame with hand gesture control
    cv2.imshow("Hand Gesture Control", frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the windows
cap.release()
cv2.destroyAllWindows()
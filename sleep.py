import cv2
import dlib
import numpy as np
import time
import winsound  # For Windows, use 'import winsound'. For Linux, use an appropriate library like 'beep'

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(dlib.shape_predictor_model)

# Define the indices of the facial landmarks for the eyes
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))

def get_eye_aspect_ratio(eye_points):
    """Calculate the Eye Aspect Ratio (EAR)"""
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

def main():
    # Set the eye aspect ratio threshold and consecutive frame count
    EAR_THRESHOLD = 0.2
    CONSECUTIVE_FRAMES = 20

    # Initialize variables
    consecutive_blinks = 0
    alarm_triggered = False

    # Start video capture
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        for face in faces:
            landmarks = predictor(gray, face)
            landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])
            
            left_eye = landmarks[LEFT_EYE]
            right_eye = landmarks[RIGHT_EYE]
            
            ear_left = get_eye_aspect_ratio(left_eye)
            ear_right = get_eye_aspect_ratio(right_eye)
            
            ear = (ear_left + ear_right) / 2.0
            
            if ear < EAR_THRESHOLD:
                consecutive_blinks += 1
                if consecutive_blinks >= CONSECUTIVE_FRAMES:
                    if not alarm_triggered:
                        print("ALARM! Driver may be drowsy.")
                        # Trigger buzzer or sound alarm
                        winsound.Beep(1000, 1000)  # Frequency, duration
                        alarm_triggered = True
            else:
                consecutive_blinks = 0
                alarm_triggered = False

        # Display the frame
        cv2.imshow("Anti-Sleep Alarm", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

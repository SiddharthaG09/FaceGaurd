import cv2
import dlib
import math
import tkinter as tk
from datetime import datetime
from playsound import playsound

def calculate_ear(landmarks, eye_indices):
    # Calculate the Euclidean distances between the vertical eye landmarks
    left_eye_width = math.dist([landmarks.part(eye_indices[0]).x, landmarks.part(eye_indices[0]).y],
                               [landmarks.part(eye_indices[3]).x, landmarks.part(eye_indices[3]).y])
    right_eye_width = math.dist([landmarks.part(eye_indices[1]).x, landmarks.part(eye_indices[1]).y],
                                [landmarks.part(eye_indices[5]).x, landmarks.part(eye_indices[5]).y])

    # Calculate the Euclidean distance between the horizontal eye landmarks
    eye_height = math.dist([landmarks.part(eye_indices[2]).x, landmarks.part(eye_indices[2]).y],
                           [landmarks.part(eye_indices[4]).x, landmarks.part(eye_indices[4]).y])

    # Calculate the EAR
    ear = (left_eye_width + right_eye_width) / (2.0 * eye_height)
    return ear

# Load the face detection model and facial landmarks predictor
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Webcam configuration
cap = cv2.VideoCapture(0)
screen_width, screen_height = 1920, 1080

# GUI window setup
root = tk.Tk()
root.title("Attention Alert")
root.geometry("300x100")
alert_label = tk.Label(root, text="", font=("Helvetica", 16))
alert_label.pack()
root.withdraw()  # Hide the window initially

EYE_AR_THRESHOLD = 0.18  # Adjusted threshold for EAR
BLINK_COUNTER_THRESHOLD = 3  # Number of consecutive frames with low EAR to detect a blink
BLINK_FRAME_THRESHOLD = 6  # Number of frames without blinking to reset blink counter

EAR_SMOOTHING_WINDOW = 5  # Number of frames for EAR smoothing

ear_history = []  # List to store recent EAR values for smoothing

blink_counter = 0  # Counter for consecutive low EAR frames (potential blink)

last_attention_time = datetime.now()
is_attention_detected = True  # Set initial state to True
gui_visible = False  # Track whether the GUI is currently visible

def play_chime():
    playsound("chime.wav")  # Replace with the path to your chime sound file

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)

    if len(faces) == 0:
        attention_status = "Not Paying Attention"
        elapsed_time = (datetime.now() - last_attention_time).seconds
        if elapsed_time >= 3 and is_attention_detected:
            alert_label.config(text="Please pay attention!", fg="red")
            root.deiconify()  # Show the GUI window
            play_chime()  # Play the chime sound
            gui_visible = True
            is_attention_detected = False
    else:
        for face in faces:
            landmarks = landmark_predictor(gray, face)

            left_eye = [(landmarks.part(36).x + landmarks.part(39).x) // 2,
                        (landmarks.part(36).y + landmarks.part(39).y) // 2]
            right_eye = [(landmarks.part(42).x + landmarks.part(45).x) // 2,
                         (landmarks.part(42).y + landmarks.part(45).y) // 2]

            # Calculate EAR for each eye
            left_ear = calculate_ear(landmarks, [36, 37, 38, 39, 40, 41])
            right_ear = calculate_ear(landmarks, [42, 43, 44, 45, 46, 47])

            # Calculate average EAR for both eyes and apply smoothing
            avg_ear = (left_ear + right_ear) / 2
            ear_history.append(avg_ear)
            if len(ear_history) > EAR_SMOOTHING_WINDOW:
                ear_history.pop(0)
            avg_ear_smoothed = sum(ear_history) / len(ear_history)

            # Check for blinks
            if avg_ear_smoothed < EYE_AR_THRESHOLD:
                blink_counter += 1
            else:
                blink_counter = 0

            if blink_counter >= BLINK_COUNTER_THRESHOLD:
                attention_status = "Blink detected"
            else:
                attention_status = "Paying attention"

                # Hide the GUI window if attention is detected
                if gui_visible:
                    alert_label.config(text="")
                    root.withdraw()
                    gui_visible = False

            if screen_width // 4 < left_eye[0] < 3 * screen_width // 4 and \
               screen_height // 4 < left_eye[1] < 3 * screen_height // 4 and \
               screen_width // 4 < right_eye[0] < 3 * screen_width // 4 and \
               screen_height // 4 < right_eye[1] < 3 * screen_height // 4:
                last_attention_time = datetime.now()  # Update the last attention time
                if not is_attention_detected:
                    is_attention_detected = True
                    if gui_visible:
                        alert_label.config(text="")
                        root.withdraw()
                        gui_visible = False
            else:
                elapsed_time = (datetime.now() - last_attention_time).seconds
                if elapsed_time >= 15 and is_attention_detected:  # Adjusted to 3 seconds
                    alert_label.config(text="Please pay attention!", fg="red")
                    root.deiconify()  # Show the GUI window
                    play_chime()  # Play the chime sound
                    gui_visible = True
                    is_attention_detected = False

    cv2.putText(frame, attention_status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Attention Detector", frame)

    if cv2.waitKey(1) == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()

# Run the GUI main loop to keep it open until manually closed
root.mainloop()

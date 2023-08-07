import cv2
import dlib
import math
import tkinter as tk
from datetime import datetime, timedelta
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

EYE_AR_THRESHOLD = 0.18
BLINK_COUNTER_THRESHOLD = 3
BLINK_FRAME_THRESHOLD = 6

EAR_SMOOTHING_WINDOW = 5

ear_history = []
blink_counter = 0
last_attention_time = datetime.now()
start_attention_time = None
best_attention_time = timedelta(seconds=-1)
total_attention_time = timedelta()
is_attention_detected = False
gui_visible = False

def play_chime():
    playsound("chime.wav")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)

    if len(faces) == 0:
        attention_status = "Not Paying Attention"
        elapsed_time = (datetime.now() - last_attention_time).seconds
        if elapsed_time >= 12 and is_attention_detected:
            alert_label.config(text="Please pay attention!", fg="red")
            root.deiconify()
            play_chime()
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

                if start_attention_time is not None:
                    current_time = datetime.now()
                    elapsed_time = (current_time - start_attention_time).total_seconds()
                    timer_text = f"Attention Timer: {int(elapsed_time)} seconds"
                    cv2.putText(frame, timer_text, (screen_width - 300, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    total_attention_time += timedelta(seconds=elapsed_time)

                    if elapsed_time > best_attention_time.total_seconds():
                        best_attention_time = timedelta(seconds=elapsed_time)

                if gui_visible:
                    alert_label.config(text="")
                    root.withdraw()
                    gui_visible = False

            if screen_width // 4 < left_eye[0] < 3 * screen_width // 4 and \
               screen_height // 4 < left_eye[1] < 3 * screen_height // 4 and \
               screen_width // 4 < right_eye[0] < 3 * screen_width // 4 and \
               screen_height // 4 < right_eye[1] < 3 * screen_height // 4:
                last_attention_time = datetime.now()
                if not is_attention_detected:
                    is_attention_detected = True
                    start_attention_time = datetime.now()
                    if gui_visible:
                        alert_label.config(text="")
                        root.withdraw()
                        gui_visible = False
            else:
                elapsed_time = (datetime.now() - last_attention_time).total_seconds()
                if elapsed_time >= 15 and is_attention_detected:
                    alert_label.config(text="Please pay attention!", fg="red")
                    root.deiconify()
                    play_chime()
                    gui_visible = True
                    is_attention_detected = False
                    start_attention_time = None

    best_text = f"Best Attention: {int(best_attention_time.total_seconds())} seconds"
    cv2.putText(frame, best_text, (20, screen_height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(frame, attention_status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Attention Detector", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Calculate total attention time in seconds
total_attention_seconds = total_attention_time.total_seconds()
best_attention_seconds = best_attention_time.total_seconds()

print("Best Attention Time:", best_attention_seconds, "seconds")
print("Total Attention Time:", total_attention_seconds, "seconds")

# Run the GUI main loop to keep it open until manually closed
root.mainloop()

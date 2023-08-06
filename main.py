import cv2
import dlib
import tkinter as tk
from tkinter import messagebox
import time

# Load the face detection model and facial landmarks predictor
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Webcam configuration
cap = cv2.VideoCapture(0)
screen_width, screen_height = 1920, 1080

attention_timer = None
attention_threshold = 15  # 15 seconds

def show_attention_popup():
    root = tk.Tk()
    root.withdraw()
    messagebox.showwarning("Get Back to Work", "You have not been paying attention. Get back to work!")
    root.destroy()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)

    attention_detected = False

    for face in faces:
        landmarks = landmark_predictor(gray, face)
        
        left_eye = [(landmarks.part(36).x + landmarks.part(39).x) // 2, (landmarks.part(36).y + landmarks.part(39).y) // 2]
        right_eye = [(landmarks.part(42).x + landmarks.part(45).x) // 2, (landmarks.part(42).y + landmarks.part(45).y) // 2]
        
        if screen_width // 3 < left_eye[0] < 2 * screen_width // 3 and \
           screen_height // 3 < left_eye[1] < 2 * screen_height // 3 and \
           screen_width // 3 < right_eye[0] < 2 * screen_width // 3 and \
           screen_height // 3 < right_eye[1] < 2 * screen_height // 3:
            attention_detected = True
            attention_status = "Paying attention"
        else:
            attention_status = "Not paying attention"

    if attention_detected:
        if attention_timer is not None:
            elapsed_time = time.time() - attention_timer
            if elapsed_time >= attention_threshold:
                show_attention_popup()
                attention_timer = None
        else:
            attention_timer = time.time()
    else:
        attention_timer = None
    
    cv2.putText(frame, attention_status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow("Attention Detector", frame)

    if cv2.waitKey(1) == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()

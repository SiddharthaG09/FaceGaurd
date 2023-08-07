import cv2
import dlib
import tkinter as tk
from datetime import datetime

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

last_attention_time = datetime.now()
is_attention_detected = True  # Set initial state to True

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)

    if len(faces) == 0:  # No faces detected
        attention_status = ""
    else:
        for face in faces:
            landmarks = landmark_predictor(gray, face)
            
            left_eye = [(landmarks.part(36).x + landmarks.part(39).x) // 2, (landmarks.part(36).y + landmarks.part(39).y) // 2]
            right_eye = [(landmarks.part(42).x + landmarks.part(45).x) // 2, (landmarks.part(42).y + landmarks.part(45).y) // 2]
            
            cv2.circle(frame, tuple(left_eye), 2, (0, 0, 255), -1)
            cv2.circle(frame, tuple(right_eye), 2, (0, 0, 255), -1)
            
            if screen_width // 3 < left_eye[0] < 2 * screen_width // 3 and \
               screen_height // 3 < left_eye[1] < 2 * screen_height // 3 and \
               screen_width // 3 < right_eye[0] < 2 * screen_width // 3 and \
               screen_height // 3 < right_eye[1] < 2 * screen_height // 3:
                attention_status = "Paying attention"
                last_attention_time = datetime.now()  # Update the last attention time
                if not is_attention_detected:
                    alert_label.config(text="")
                    root.withdraw()  # Hide the GUI window if attention is detected
                    is_attention_detected = True
            else:
                attention_status = "Not paying attention"
                elapsed_time = (datetime.now() - last_attention_time).seconds
                if elapsed_time >= 15 and is_attention_detected:
                    alert_label.config(text="Please pay attention!", fg="red")
                    root.deiconify()  # Show the GUI window
                    is_attention_detected = False
        
    cv2.putText(frame, attention_status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Attention Detector", frame)

    if cv2.waitKey(1) == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()

# Run the GUI main loop to keep it open until manually closed
root.mainloop()


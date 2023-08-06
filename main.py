import cv2
import dlib

# Load the face detection model and facial landmarks predictor
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Webcam configuration
cap = cv2.VideoCapture(0)
screen_width, screen_height = 1920, 1080

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)

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
        else:
            attention_status = "Not paying attention"
        
        cv2.putText(frame, attention_status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow("Attention Detector", frame)

    if cv2.waitKey(1) == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()

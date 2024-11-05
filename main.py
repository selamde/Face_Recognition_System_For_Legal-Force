import cv2
import os
from deepface import DeepFace

# Ensure the folder for saving screenshots exists
if not os.path.exists('criminal_face'):
    os.makedirs('criminal_face')

# Load the pre-trained face detector model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0
face_match = False
screenshot_taken = False  # New flag to control screenshot capture
reference_img = cv2.imread("selamsew.jpg")

def check_face(frame):
    global face_match
    try:
        if DeepFace.verify(frame, reference_img.copy(), model_name="Facenet")['verified']:
            face_match = True
        else:
            face_match = False
    except ValueError:
        face_match = False

while True:
    ret, frame = cap.read()

    if ret:
        # Resize the frame for faster processing
        resized_frame = cv2.resize(frame, (240, 180))

        # Convert the frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # If there is a face match and a screenshot has not been taken yet, save the face
            if face_match and not screenshot_taken:
                face_crop = frame[y:y + h, x:x + w]
                filename = f'criminal_face/match_{counter}.jpg'
                cv2.imwrite(filename, face_crop)
                print(f"Saved screenshot: {filename}")
                screenshot_taken = True  # Set flag to avoid multiple saves

        # Check face every 60 frames
        if counter % 60 == 0:
            check_face(resized_frame.copy())  # Remove threading for better performance on CPU

        counter += 1

        # Display result
        if face_match:
            cv2.putText(frame, "MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "NO MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            screenshot_taken = False  # Reset flag when there is no match

        cv2.imshow("video", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

import cv2

# Load the trained face recognizer from the file
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_recognition_model.yml")

# Initialize the video capture device
cap = cv2.VideoCapture(0)

# Define the face detection classifier
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
if face_cascade.empty():
    print("Error: Could not load the face cascade classifier.")
    exit()

while True:
    # Capture a frame from the video feed
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    # Loop through each detected face
    for (x, y, w, h) in faces:
        # Extract the face ROI from the grayscale frame
        face_roi = gray[y:y+h, x:x+w]

        # Use the face recognizer to predict the label of the face ROI
        predicted_label, confidence = face_recognizer.predict(face_roi)

        # Draw a rectangle around the detected face
        color = (0, 255, 0)
        thickness = 2
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)

        # Add the predicted label and confidence to the face rectangle
        label_text = "Person {}".format(predicted_label)
        label_position = (x, y-10)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        cv2.putText(frame, label_text, label_position, font, font_scale, color, thickness)

    # Display the frame with detected faces
    cv2.imshow("Face Recognition", frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture device and close the windows
cap.release()
cv2.destroyAllWindows()

import cv2

# Load the trained face recognizer from the file
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_recognition_model.yml")

# Load a test image and convert it to grayscale
test_image = cv2.imread("test_image.jpg")
if test_image is not None:
    gray_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

    # Use the face recognizer to predict the label of the test image
    predicted_label, confidence = face_recognizer.predict(gray_image)

    # Print the predicted label and confidence
    print("Predicted label: {}".format(predicted_label))
    print("Confidence: {}".format(confidence))
else:
    print("Error: Could not load the test image.")

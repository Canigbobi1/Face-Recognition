import cv2
import os
import numpy as np

# Initialize LBPH face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Define the path to the dataset directory
dataset_path = "./dataset"

# Initialize lists to store the face images and labels
face_images = []
labels = []

# Loop through each subdirectory in the dataset directory
for subdirectory in os.listdir(dataset_path):

    # Define the path to the subdirectory
    subdirectory_path = os.path.join(dataset_path, subdirectory)

    # Loop through each image file in the subdirectory
    for filename in os.listdir(subdirectory_path):

        # Define the path to the image file
        filepath = os.path.join(subdirectory_path, filename)

        # Load the image as grayscale
        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

        # Add the image and label to the lists
        face_images.append(image)
        labels.append(int(subdirectory))

# Train the face recognizer on the face images and labels
face_recognizer.train(face_images, np.array(labels))

# Save the trained model to a file
face_recognizer.save("face_recognition_model.yml")

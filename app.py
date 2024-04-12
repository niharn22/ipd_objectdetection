import cv2
import math
import numpy as np
import streamlit as st
import pygame

# Initialize Pygame mixer
pygame.mixer.init()

# Define the calculate_distance function
def calculate_distance(face_width, focal_length, actual_width):
    return (actual_width * focal_length) / face_width

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Set the actual width of the face (in centimeters)
actual_width = 15.0  # You can adjust this value based on your specific setup and measurements

# Set the focal length (in pixels) of the camera
# To find the focal length, you can refer to camera calibration techniques
# or use a rough estimation based on the camera specifications
focal_length = 1000.0  # You need to adjust this value based on your camera's specifications

st.title("Distance Measurement App")

st.markdown("This app measures the distance from the camera to detected faces.")

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("Error: Unable to open camera.")

# Create a placeholder for the video stream
video_placeholder = st.empty()

# Load the beep sound
pygame.mixer.music.load("beep.mp3")

danger_zone = False

while cap.isOpened():
    # Read the frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    in_danger_zone = False

    # Process each detected face
    for (x, y, w, h) in faces:
        # Calculate the distance from the camera to the detected face
        distance = calculate_distance(w, focal_length, actual_width)

        # Display the distance on the frame
        cv2.putText(frame, f"Distance: {distance:.2f} cm", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Check if the person is in the danger zone
        if distance < 80:
            cv2.putText(frame, "DANGER!", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            in_danger_zone = True

    # If the person is in the danger zone and the sound is not already playing, play the sound
    if in_danger_zone and not danger_zone:
        pygame.mixer.music.play(-1)  # -1 means loop the sound indefinitely
        danger_zone = True
    # If the person leaves the danger zone, stop playing the sound
    elif not in_danger_zone and danger_zone:
        pygame.mixer.music.stop()
        danger_zone = False

    # Display the result
    video_placeholder.image(frame, channels="BGR", caption='Distance Measurement')

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()

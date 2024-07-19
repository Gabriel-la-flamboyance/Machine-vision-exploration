import numpy as np
import cv2
import time

# Path to the directory containing the model
path_model = "C:/Users/mugis/Documents/Master_IN/visionBasedEmbeddesSystem/recap/models/"

# Name of the model file
model_name = "model-f6b98070.onnx"

# Load the neural network model using OpenCV's DNN module
model = cv2.dnn.readNet(path_model + model_name)

# Check if the model was loaded successfully
if model.empty():
    print("Could not load the neural net! - Check path")

# Set the preferable backend and target to CUDA for GPU acceleration
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Open the webcam (device 0)
cap = cv2.VideoCapture(0)

# Continuously capture frames from the webcam
while cap.isOpened():
    # Read a frame from the webcam
    success, img = cap.read()

    # Get the dimensions of the image
    imgHeight, imgWidth, channels = img.shape

    # Start time to calculate FPS
    start = time.time()

    # Convert the image from BGR to RGB color space
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Create a blob from the image
    blob = cv2.dnn.blobFromImage(img, 1/255., (384, 384), (123.675, 116.28, 103.53), True, False)

    # Set the blob as input to the model
    model.setInput(blob)

    # Perform a forward pass through the network
    output = model.forward()

    # Process the output to get the desired shape and size
    output = output[0, :, :]
    output = cv2.resize(output, (imgWidth, imgHeight))

    # Normalize the output to range [0, 1]
    output = cv2.normalize(output, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # End time to calculate FPS
    end = time.time()
    # Calculate the FPS for the current frame
    fps = 1 / (end - start)

    # Display the FPS on the image
    cv2.putText(img, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Convert the image back to BGR color space for display
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Display the original image
    cv2.imshow('image', img)
    # Display the processed output (Depth Map)
    cv2.imshow('Depth Map', output)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

import cv2
import torch
import numpy as np
import os
from ultralytics import YOLO

# Define the path to the YOLOv8n-face model
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
MODEL_FILE = os.path.join(MODEL_DIR, 'yolov8n-face.pt')

# Ensure the model file exists
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError("YOLOv8n-face model not found in 'models/' directory. Please check the file path.")

# Load the YOLOv8 model
model = YOLO(MODEL_FILE)

def main():
    # Capture video from the webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Mirror the frame for a more natural view
        frame = cv2.flip(frame, 1)
        
        # Run YOLOv8 face detection
        results = model(frame)
        
        # Process detected faces
        for result in results:
            for box in result.boxes.data:
                x, y, x1, y1, confidence = box[:5]  # Extract bounding box and confidence
                x, y, x1, y1 = int(x), int(y), int(x1), int(y1)
                
                # Draw rectangle around detected face
                cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
                label = f"Face: {confidence*100:.2f}%"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        
        # Display the frame
        cv2.imshow('YOLOv8 Face Detection', frame)
        
        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
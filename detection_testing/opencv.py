import cv2
import numpy as np
import os

# Define the correct path to the models folder
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
MODEL_FILE = os.path.join(MODEL_DIR, 'opencv_face_detector_uint8.pb')
CONFIG_FILE = os.path.join(MODEL_DIR, 'opencv_face_detector.pbtxt')

# Ensure model files exist
if not os.path.exists(MODEL_FILE) or not os.path.exists(CONFIG_FILE):
    raise FileNotFoundError("Model files not found in the 'models/' directory. Please check the file paths.")

# Load the DNN model
net = cv2.dnn.readNetFromTensorflow(MODEL_FILE, CONFIG_FILE)

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
        
        # Get frame dimensions
        h, w = frame.shape[:2]
        
        # Prepare input blob for the DNN model
        blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0), swapRB=False, crop=False)
        
        # Set the input to the model
        net.setInput(blob)
        
        # Perform face detection
        detections = net.forward()
        
        # Process detected faces
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # Confidence threshold
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                
                # Draw rectangle around detected face
                cv2.rectangle(frame, (x, y), (x1, y1), (255, 0, 0), 2)
                label = f"Face: {confidence*100:.2f}%"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
        
        # Display the frame
        cv2.imshow('Real-Time Face Detection (DNN)', frame)
        
        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

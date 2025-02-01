from config import FACE_DB_PATH, YOLO_MODEL_PATH
from utils.face_db import build_face_db
from utils.helpers import validate_new_person, save_uploaded_files
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import pickle
import time

# Initialize models once
@st.cache_resource
def load_models():
    try:
        face_model = YOLO(str(YOLO_MODEL_PATH))
        with open(FACE_DB_PATH, 'rb') as f:
            face_db = pickle.load(f)
        return face_model, face_db
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

face_model, face_db = load_models()

def recognize_face(face_img):
    try:
        result = DeepFace.represent(
            face_img,
            model_name='Facenet',
            detector_backend='retinaface',
            enforce_detection=False
        )
        
        if not result:
            return "Unknown"
            
        query_embedding = result[0]["embedding"]
        min_dist = float('inf')
        identity = "Unknown"
        
        for name, embeddings in face_db.items():
            for emb in embeddings:
                dist = np.linalg.norm(np.array(emb) - np.array(query_embedding))
                if dist < min_dist and dist < 0.65:  # Adjusted threshold
                    min_dist = dist
                    identity = name
                    
        return identity
    except Exception as e:
        print(f"Recognition error: {str(e)}")
        return "Unknown"

def main():
    st.title("Robust Face Recognition System")
    
    # Sidebar with improved UI
    with st.sidebar:
        st.header("Manage Known Faces")
        new_name = st.text_input("Full Name")
        uploaded_files = st.file_uploader(
            "Upload 3+ Face Images (front-facing, good lighting)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )
        
        if st.button("Add New Person"):
            if validate_new_person(new_name, uploaded_files):
                try:
                    save_path = save_uploaded_files(new_name, uploaded_files)
                    if build_face_db():
                        # Reload face database
                        global face_db
                        with open(FACE_DB_PATH, 'rb') as f:
                            face_db = pickle.load(f)
                        st.success(f"Successfully added {new_name}!")
                        st.rerun()
                    else:
                        st.error("Failed to update face database")
                except Exception as e:
                    st.error(f"Error adding person: {str(e)}")
    
    # Main webcam interface
    st.header("Live Recognition")
    run = st.checkbox("Start Camera")
    frame_placeholder = st.empty()
    
    cap = cv2.VideoCapture(0)
    
    while run:
        success, frame = cap.read()
        if not success:
            st.error("Failed to access camera")
            break
            
        # YOLO Face Detection
        results = face_model(frame, verbose=False, imgsz=320)
        
        # Process detections
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                face_img = frame[y1:y2, x1:x2]
                
                # Skip small faces
                if (x2 - x1) < 50 or (y2 - y1) < 50:
                    continue
                
                # Convert BGR to RGB for recognition
                face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                
                # Recognition
                identity = recognize_face(face_img_rgb)
                
                # Draw UI
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, identity, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Display frame
        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
    cap.release()

if __name__ == "__main__":
    main()
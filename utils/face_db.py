from config import KNOWN_FACES_DIR, FACE_DB_PATH
from deepface import DeepFace
import cv2
import pickle
import os
import numpy as np
from tqdm import tqdm

def build_face_db():
    """Robust face database builder with error handling"""
    face_db = {}
    
    try:
        for person_dir in tqdm(list(KNOWN_FACES_DIR.iterdir()), desc="Processing people"):
            if not person_dir.is_dir():
                continue
                
            embeddings = []
            person_name = person_dir.name
            
            for img_path in person_dir.glob("*"):
                if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                    continue
                
                try:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                        
                    # Get face embedding with error handling
                    result = DeepFace.represent(
                        img_path,
                        model_name='Facenet',
                        detector_backend='retinaface',
                        enforce_detection=True
                    )
                    
                    if isinstance(result, list):
                        embedding = result[0]["embedding"]
                        embeddings.append(embedding)
                        
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")
                    continue
            
            if embeddings:
                face_db[person_name] = embeddings
                
        with open(FACE_DB_PATH, 'wb') as f:
            pickle.dump(face_db, f)
            
        return True
    
    except Exception as e:
        print(f"Critical error building face DB: {str(e)}")
        return False
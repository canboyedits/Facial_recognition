import cv2
import os
import dlib
import sqlite3
import numpy as np
from ultralytics import YOLO
import tkinter as tk
from tkinter import simpledialog, messagebox, filedialog
from keras.models import load_model

# Define paths
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
YOLO_MODEL_FILE = os.path.join(MODEL_DIR, 'yolov8n-face.pt')
DLIB_FACE_REC_MODEL = os.path.join(MODEL_DIR, 'dlib_face_recognition_resnet_model_v1.dat')
DLIB_LANDMARKS_MODEL = os.path.join(MODEL_DIR, 'shape_predictor_68_face_landmarks.dat')
EMOTION_MODEL_FILE = os.path.join(MODEL_DIR, 'emotion_model.hdf5')
DB_FILE = "face_data.db"

# Load YOLOv8 model
if not os.path.exists(YOLO_MODEL_FILE):
    raise FileNotFoundError("YOLOv8n-face model not found in 'models/' directory.")
model = YOLO(YOLO_MODEL_FILE)

# Load Dlib models
if not os.path.exists(DLIB_FACE_REC_MODEL) or not os.path.exists(DLIB_LANDMARKS_MODEL):
    raise FileNotFoundError("Dlib models not found in 'models/' directory.")
face_rec_model = dlib.face_recognition_model_v1(DLIB_FACE_REC_MODEL)
shape_predictor = dlib.shape_predictor(DLIB_LANDMARKS_MODEL)
detector = dlib.get_frontal_face_detector()

# ---------------------------
# Load Emotion Model
# ---------------------------
if not os.path.exists(EMOTION_MODEL_FILE):
    raise FileNotFoundError("Emotion model not found in './models/' directory.")

emotion_model = load_model(EMOTION_MODEL_FILE)

# Edit these labels according to your own model
emotion_labels = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]


def get_emotion(face_img):
    """Predict emotion from face crop"""
    try:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (48, 48))
        gray = gray.astype("float") / 255.0
        gray = np.expand_dims(gray, axis=-1)
        gray = np.expand_dims(gray, axis=0)

        preds = emotion_model.predict(gray, verbose=0)
        emotion_id = np.argmax(preds[0])
        return emotion_labels[emotion_id]
    except:
        return "Unknown"


def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""CREATE TABLE IF NOT EXISTS faces (
                      id INTEGER PRIMARY KEY AUTOINCREMENT,
                      name TEXT,
                      encoding BLOB)
                   """)
    conn.commit()
    conn.close()

init_db()


def get_face_encoding(image):
    """
    Extract face encoding using dlib
    """
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector(rgb_image)

    if len(faces) == 0:
        return None
    
    shape = shape_predictor(rgb_image, faces[0])
    face_descriptor = face_rec_model.compute_face_descriptor(rgb_image, shape)
    encoding = np.array(face_descriptor)
    
    return encoding


def add_face():
    file_path = filedialog.askopenfilename(
        title="Select Face Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    
    if not file_path:
        messagebox.showerror("Error", "No file selected.")
        return
    
    image = cv2.imread(file_path)
    if image is None:
        messagebox.showerror("Error", "Failed to load the selected image.")
        return
    
    roi = cv2.selectROI("Select Face Region", image, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Face Region")
    
    if roi == (0, 0, 0, 0):
        messagebox.showerror("Error", "No region selected.")
        return
    
    x, y, w, h = roi
    face_crop = image[y:y+h, x:x+w]
    
    if face_crop.size == 0:
        messagebox.showerror("Error", "Selected region is too small.")
        return
    
    encoding = get_face_encoding(face_crop)
    
    if encoding is None:
        messagebox.showerror("Error", "No face detected in the selected region.")
        return
    
    name = simpledialog.askstring("Input", "Enter the person's name:")
    if not name:
        messagebox.showerror("Error", "Name not entered.")
        return
    
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO faces (name, encoding) VALUES (?, ?)", (name, encoding.tobytes()))
    conn.commit()
    conn.close()
    messagebox.showinfo("Success", f"Face for {name} added successfully!")


def recognize_faces():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT name, encoding FROM faces")
    stored_faces = cursor.fetchall()
    conn.close()
    
    if not stored_faces:
        messagebox.showinfo("Info", "No faces in the database. Please add faces first.")
        return
    
    known_encodings = []
    known_names = []
    
    for name, encoding_blob in stored_faces:
        encoding = np.frombuffer(encoding_blob, dtype=np.float64)
        known_encodings.append(encoding)
        known_names.append(name)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open webcam.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        results = model(frame)
        
        for result in results:
            for box in result.boxes.data:
                x, y, x1, y1, confidence = box[:5]
                x, y, x1, y1 = int(x), int(y), int(x1), int(y1)
                
                if x >= 0 and y >= 0 and x1 < frame.shape[1] and y1 < frame.shape[0]:
                    face_crop = frame[y:y1, x:x1]
                    
                    if face_crop.shape[0] < 20 or face_crop.shape[1] < 20:
                        continue
                    
                    # Face Recognition
                    encoding = get_face_encoding(face_crop)
                    
                    if encoding is not None:
                        distances = [np.linalg.norm(encoding - known) for known in known_encodings]
                        
                        if distances:
                            min_dist = min(distances)
                            if min_dist < 0.6:
                                name = known_names[distances.index(min_dist)]
                            else:
                                name = "Unknown"
                        else:
                            name = "Unknown"

                        # Emotion Detection
                        emotion = get_emotion(face_crop)

                        # Final label with emotion
                        label = f"{name} | {emotion}"

                        cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        
        cv2.imshow("Face Recognition with Emotion", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def create_gui():
    root = tk.Tk()
    root.title("Face Recognition System")
    root.geometry("400x300")
    
    tk.Label(root, text="Face Recognition System", font=("Arial", 16)).pack(pady=10)
    
    tk.Button(root, text="Add Face", command=add_face, font=("Arial", 12)).pack(pady=10)
    tk.Button(root, text="Recognize Faces", command=recognize_faces, font=("Arial", 12)).pack(pady=10)
    tk.Button(root, text="Exit", command=root.quit, font=("Arial", 12)).pack(pady=10)
    
    root.mainloop()


if __name__ == "__main__":
    create_gui()

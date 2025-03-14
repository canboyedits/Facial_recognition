import pickle
from config import FACE_DB_PATH

with open(FACE_DB_PATH, 'rb') as f:
    face_db = pickle.load(f)

print("Stored Face Embeddings:")
for name, embeddings in face_db.items():
    print(f"{name}: {len(embeddings)} embeddings")
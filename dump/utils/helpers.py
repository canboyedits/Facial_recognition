from config import KNOWN_FACES_DIR, MODELS_DIR
import shutil
import streamlit as st

def validate_new_person(name, files):
    """Validate new person inputs"""
    if not name:
        st.error("Name cannot be empty!")
        return False
    if len(files) < 3:
        st.error("At least 3 images required!")
        return False
    if (KNOWN_FACES_DIR / name).exists():
        st.error(f"Person '{name}' already exists!")
        return False
    return True

def save_uploaded_files(name, files):
    """Save uploaded files to known_faces directory"""
    person_dir = KNOWN_FACES_DIR / name
    person_dir.mkdir(exist_ok=True)
    
    # Clear existing files if any
    for existing_file in person_dir.glob("*"):
        existing_file.unlink()
    
    # Save new files
    for i, file in enumerate(files):
        with open(person_dir / f"image_{i+1}.jpg", "wb") as f:
            f.write(file.getbuffer())
    
    return str(person_dir)
import os

# Suppress TF/absl log spam before DeepFace imports TensorFlow.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np
from deepface import DeepFace

class FaceIdentifier:
    def __init__(self, known_faces_dir="known_managers"):
        self.known_faces_dir = os.path.abspath(known_faces_dir)
        
        if not os.path.exists(self.known_faces_dir):
            os.makedirs(self.known_faces_dir)
            print(f"[FACE ID] Created directory: {self.known_faces_dir}")
            print(f"[FACE ID] Please add manager photos (e.g., John.jpg) to this folder.")
        
        photos = self._get_photo_list()
        if photos:
            print(f"[FACE ID] Loaded {len(photos)} manager photo(s): {photos}")
        else:
            print(f"[FACE ID] WARNING: No manager photos found in: {self.known_faces_dir}")
            print(f"[FACE ID] System will detect managers by YOLO class, but cannot identify names.")
            
    def _get_photo_list(self):
        """Returns list of valid image files in the known_faces_dir."""
        if not os.path.exists(self.known_faces_dir):
            return []
        return [f for f in os.listdir(self.known_faces_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    
    def identify_face(self, frame_crop):
        """
        Attempts to identify the cropped frame against images in known_managers.
        Returns the name of the manager if matched, otherwise None.
        Returns None immediately if no photos are available.
        """
        if not self._get_photo_list():
            return None  # No reference photos to compare against
        
        try:
            # enforce_detection=False: don't crash if crop has no clear face (e.g., side profile)
            # silent=True: suppress DeepFace's own print statements
            dfs = DeepFace.find(
                img_path=frame_crop,
                db_path=self.known_faces_dir,
                enforce_detection=False,
                silent=True
            )
            
            if len(dfs) > 0 and not dfs[0].empty:
                # Get the best (closest) match
                best_row = dfs[0].iloc[0]
                match_path = best_row['identity']
                # Use filename (without extension) as the manager's name
                name = os.path.splitext(os.path.basename(match_path))[0]
                print(f"[FACE ID] Matched face to: {name}")
                return name
                
        except Exception as e:
            # Silently handle errors: bad crop, no face detected, etc.
            pass
            
        return None

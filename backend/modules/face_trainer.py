"""Utility functions for training face recognition models from stored images."""
from __future__ import annotations

import base64
import os
from datetime import datetime
from typing import Dict, List, Tuple

import cv2
import numpy as np

from database import get_face_images_collection, get_face_embeddings_collection

# the backend/main.py already inserts src directory onto sys.path
# so we can import vision and face_db from the original library if needed
try:
    from src.vision import get_analyzer
except ImportError:
    # older path without src prefix
    from vision import get_analyzer


def _decode_image(base64_str: str) -> np.ndarray:
    """Decode a base64-encoded JPEG/PNG to a BGR numpy array."""
    data = base64.b64decode(base64_str)
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Unable to decode image")
    return img


def train_person_from_images(person_name: str) -> Tuple[int, int, List[str]]:
    """Convert stored face images into embeddings for a person.

    Returns a tuple (successful_count, failed_count, errors).
    """
    images_coll = get_face_images_collection()
    emb_coll = get_face_embeddings_collection()

    images = list(images_coll.find({"person_name": person_name}))
    successful = 0
    failed = 0
    errors: List[str] = []

    # build analyzer with config from environment
    use_gpu = os.getenv("FACE_REG_USE_GPU", "false").lower() == "true"
    analyzer = get_analyzer(use_gpu=use_gpu)

    for img_doc in images:
        try:
            img = _decode_image(img_doc["image_base64"])
            faces = analyzer.get(img)
            if not faces:
                raise RuntimeError("No face detected")
            # pick first face
            face = faces[0]
            emb = face.normed_embedding
            # store embedding in db
            emb_coll.insert_one({
                "person_name": person_name,
                "embedding": emb.tolist(),
                "created_at": datetime.utcnow(),
            })
            # mark image as trained
            images_coll.update_one({"_id": img_doc["_id"]}, {"$set": {"trained": True}})
            successful += 1
        except Exception as e:
            failed += 1
            errors.append(str(e))

    return successful, failed, errors


def get_training_status(person_name: str) -> Dict[str, int]:
    """Return counts of images and embeddings for a person."""
    images_coll = get_face_images_collection()
    emb_coll = get_face_embeddings_collection()

    total_images = images_coll.count_documents({"person_name": person_name})
    trained_images = images_coll.count_documents({"person_name": person_name, "trained": True})
    embedding_count = emb_coll.count_documents({"person_name": person_name})

    return {
        "total_images": total_images,
        "trained_images": trained_images,
        "embedding_count": embedding_count,
    }


def retrain_person(person_name: str) -> Tuple[int, int, List[str]]:
    """Delete existing embeddings and retrain from scratch."""
    emb_coll = get_face_embeddings_collection()
    emb_coll.delete_many({"person_name": person_name})
    # mark images as untrained
    images_coll = get_face_images_collection()
    images_coll.update_many({"person_name": person_name}, {"$set": {"trained": False}})

    return train_person_from_images(person_name)


def delete_person_training_data(person_name: str) -> Dict[str, int]:
    """Erase all stored images and embeddings for a person."""
    images_coll = get_face_images_collection()
    emb_coll = get_face_embeddings_collection()
    images_deleted = images_coll.delete_many({"person_name": person_name}).deleted_count
    embeddings_deleted = emb_coll.delete_many({"person_name": person_name}).deleted_count
    return {"images_deleted": images_deleted, "embeddings_deleted": embeddings_deleted}

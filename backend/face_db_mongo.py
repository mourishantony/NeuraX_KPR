"""Legacy helpers for interacting with face-related data in MongoDB."""
from __future__ import annotations

from database import get_face_images_collection, get_face_embeddings_collection


def get_all_registered_names():
    """Return a sorted list of unique person names that have face data."""
    imgs = get_face_images_collection()
    names = imgs.distinct("person_name")
    return sorted(names)

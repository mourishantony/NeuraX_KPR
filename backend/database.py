"""MongoDB database connection and initialization."""
from __future__ import annotations

import os
from typing import Optional

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from dotenv import load_dotenv

# Load environment variables
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env", override=False)

# Global connection objects
_client: Optional[MongoClient] = None
_db: Optional[Database] = None


def get_mongo_client() -> MongoClient:
    """Get or create MongoDB client."""
    global _client
    if _client is None:
        mongo_uri = os.getenv(
            "MONGODB_URI",
            "mongodb://localhost:27017"
        )
        _client = MongoClient(mongo_uri)
        # Test connection
        try:
            _client.admin.command('ping')
            print("[Database] Connected to MongoDB successfully")
        except Exception as e:
            print(f"[Database] Warning: Could not connect to MongoDB: {e}")
            print("[Database] Using memory-based storage fallback")
    return _client


def get_database() -> Database:
    """Get database instance."""
    global _db
    if _db is None:
        client = get_mongo_client()
        db_name = os.getenv("MONGODB_DB_NAME", "neurax_kpr")
        _db = client[db_name]
    return _db


def get_users_collection() -> Collection:
    """Get users collection."""
    db = get_database()
    collection = db["users"]
    # Create index on username for faster lookups
    collection.create_index("username", unique=True)
    return collection


def get_persons_collection() -> Collection:
    """Get registered persons collection."""
    db = get_database()
    return db["persons"]


def get_alerts_collection() -> Collection:
    """Get alerts collection."""
    db = get_database()
    return db["alerts"]


def get_contacts_collection() -> Collection:
    """Get contacts tracing collection."""
    db = get_database()
    return db["contacts"]


def get_mdr_collection() -> Collection:
    """Get MDR (Multidrug Resistant) data collection."""
    db = get_database()
    return db["mdr"]


def get_pathogens_collection() -> Collection:
    """Get pathogens collection."""
    db = get_database()
    return db["pathogens"]


def get_mdr_patients_collection() -> Collection:
    """Get MDR patients collection."""
    db = get_database()
    return db["mdr_patients"]


def get_face_embeddings_collection() -> Collection:
    """Get face embeddings collection."""
    db = get_database()
    return db["face_embeddings"]


def get_face_images_collection() -> Collection:
    """Get face images metadata collection."""
    db = get_database()
    return db["face_images"]


def get_unknown_contacts_collection() -> Collection:
    """Get unknown contacts collection."""
    db = get_database()
    return db["unknown_contacts"]


def generate_person_id() -> str:
    """Generate a unique person ID."""
    import uuid
    return str(uuid.uuid4())


def initialize_database() -> None:
    """Initialize database and create default indexes."""
    try:
        db = get_database()
        
        # Initialize all collections and create indexes
        users = get_users_collection()
        persons = get_persons_collection()
        alerts = get_alerts_collection()
        contacts = get_contacts_collection()
        mdr = get_mdr_collection()
        pathogens = get_pathogens_collection()
        mdr_patients = get_mdr_patients_collection()
        face_embeddings = get_face_embeddings_collection()
        face_images = get_face_images_collection()
        unknown_contacts = get_unknown_contacts_collection()
        
        # Create indexes for better query performance
        users.create_index("username", unique=True)
        users.create_index("email", unique=True)
        
        persons.create_index("name")
        persons.create_index("phone", unique=True, sparse=True)
        persons.create_index("person_id", unique=True)
        
        alerts.create_index("created_at", background=True)
        alerts.create_index("person_id", background=True)
        
        contacts.create_index("person_id", background=True)
        contacts.create_index("contact_person_id", background=True)
        
        mdr.create_index("pathogen_id", background=True)
        mdr_patients.create_index("person_id", background=True)
        
        pathogens.create_index("name", unique=True)
        
        face_embeddings.create_index("person_id", background=True)
        face_images.create_index("person_id", background=True)
        
        unknown_contacts.create_index("created_at", background=True)
        
        print("[Database] Database initialized successfully")
    except Exception as e:
        print(f"[Database] Error initializing database: {e}")
        raise


def close_connection() -> None:
    """Close MongoDB connection."""
    global _client, _db
    if _client is not None:
        _client.close()
        _client = None
        _db = None
        print("[Database] Database connection closed")

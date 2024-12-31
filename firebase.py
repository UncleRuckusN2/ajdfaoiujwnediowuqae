# firebase.py
import os
import json
from firebase_admin import credentials, firestore, initialize_app
import logging

logger = logging.getLogger(__name__)

try:
    if os.environ.get('FIREBASE_CREDENTIALS'):
        cred = credentials.Certificate(json.loads(os.environ.get('FIREBASE_CREDENTIALS')))
    else:
        cred = credentials.Certificate("./config/solwind-3e0d2-firebase-adminsdk-xithf-a6cb38771f.json")
    initialize_app(cred)
    db = firestore.client()
    logger.info("Firebase initialized successfully")
except Exception as e:
    logger.error(f"Firebase initialization error: {str(e)}")
    raise

# Expose `db` for other modules

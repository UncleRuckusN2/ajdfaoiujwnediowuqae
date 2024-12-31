from flask import Blueprint, jsonify, request
import uuid
from firebase import db
from datetime import datetime

share_bp = Blueprint('share', __name__)

@share_bp.route('/api/share', methods=['POST'])
def create_share():
    """Create a new shared story."""
    try:
        # Generate a unique ID for the share
        share_id = str(uuid.uuid4())[:8]
        
        # Get the data from request
        data = request.get_json()
        
        # Add metadata
        share_data = {
            'data': data,
            'tier_info': data.get('tier_info'),  # e.g. pass in from frontend
            'created_at': datetime.utcnow().isoformat(),
            'wallet': data.get('wallet', 'anonymous')
        }
        
        # Store in Firebase
        db.collection('shares').document(share_id).set(share_data)
        
        return jsonify({
            'success': True,
            'share_id': share_id
        })
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@share_bp.route('/api/share/<share_id>', methods=['GET'])
def get_share(share_id):
    """Retrieve a shared story."""
    try:
        # Get from Firebase
        doc = db.collection('shares').document(share_id).get()
        
        if not doc.exists:
            return jsonify({
                'error': 'Share not found'
            }), 404
            
        share_data = doc.to_dict()
        
        return jsonify({
            'success': True,
            'data': share_data
        })
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500
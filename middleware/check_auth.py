from functools import wraps
from flask import request, jsonify
from utils.jwt_utils import verify_access_token

def check_auth(required_role=None):
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            token_header = request.headers.get("Authorization")
           
            if not token_header:
                return jsonify({"error": "Token is missing"}), 400

            try:
                token = token_header.split("Bearer")[1].strip()
            except IndexError:
                return jsonify({"error": "Invalid token format. Use 'Bearer <token>'"}), 400

            # ✅ Unpack the tuple
            error, payload = verify_access_token(token)

            if error:
                return jsonify({"error": error}), 401

            # ✅ payload is now a dict
            if required_role and payload.get("role") != required_role:
                return jsonify({"error": "Unauthorized access"}), 403

            request.user = payload  
           
            # Attach user info to request
            return func(*args, **kwargs)
        return wrapper
    return decorator

from datetime import datetime,timedelta
import os 
import jwt



access_token=os.environ.get("SECRET_KEY")


def generate_access_token(payload:dict):
    payload.update({
        
        "exp":datetime.utcnow()+timedelta(hours=8),
        "iat":datetime.utcnow()
    })
    
    return jwt.encode(payload,access_token,algorithm="HS256")


def verify_access_token(token:str):
    try:
        # payload=jwt.decode(token,access_token,algorithm=["HS256"])
        payload = jwt.decode(token, access_token, algorithms=["HS256"]) 
        return None,payload
    except Exception as e:
        return str(e),None




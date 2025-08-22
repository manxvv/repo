import pymongo
import os
from dotenv import load_dotenv



load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")

client = pymongo.MongoClient(MONGO_URI)
db = client[DB_NAME]

def get_db():
    return db
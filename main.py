import os
from flask import Flask, request, jsonify ,send_from_directory
import fitz  # PyMuPDF
from PIL import Image
from ultralytics import YOLO
import cv2
from collections import Counter
from flask_cors import CORS
import shutil
from pdfextractdata import scan_pdf
from db import get_db
import bcrypt
import jwt
import datetime
from dotenv import load_dotenv
from test import mapping

app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
db = get_db()
users_collection=db["users"]

CORS(
     app,
    supports_credentials=True,
    origins="*",
    allow_headers=["Content-Type", "Authorization"],
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"]
)

SECRET_KEY = os.getenv("SECRET_KEY")


# Folder to save uploaded PDFs
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
OUTPUT_FOLDER = "processed_images"
OUTPUT="output"
INPUT="input"
IMAGES_FOLDER = "images_by_sector"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === Load YOLO model once ===
model = YOLO("best.pt")
class_names = ["caution", "warning", "notice"]


app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # limit: 16MB




@app.route("/signup", methods=["POST"])
def signup():
    data = request.json
    username = data.get("username")
    password = data.get("password")
    role = data.get("role", "admin")

    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400

    if users_collection.find_one({"username": username}):
        return jsonify({"error": "User already exists"}), 400

    # Hash password as UTF-8 string (not binary)
    hashed_pw = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    users_collection.insert_one({
        "username": username,
        "password": hashed_pw,
        "role": role
    })

    return jsonify({"message": "User created successfully"}), 201

@app.route("/login", methods=["POST"])
def login():
    data = request.json
    username = data.get("username")
    password = data.get("password")

    user = users_collection.find_one({"username": username})
    if not user:
        return jsonify({"error": "Invalid username or password"}), 401

    if bcrypt.checkpw(password.encode("utf-8"), user["password"].encode("utf-8")):
        # Create JWT token
        payload = {
            "username": username,
            "role": user.get("role", "user"),
            "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1)  # expires in 1 hour
        }
        token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")

        return jsonify({
            "message": "Login successful",
            "username": username,
            "role": user.get("role", "user"),
            "access_token": token
        }), 200
    else:
        return jsonify({"error": "Invalid username or password"}), 401


@app.route("/extract-page", methods=["POST"])
def extract_page():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        uploaded_file = request.files["file"]

        if uploaded_file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        temp_path = os.path.join("uploads", uploaded_file.filename)
        os.makedirs("uploads", exist_ok=True)
        uploaded_file.save(temp_path)

        fullpath=os.path.join(os.getcwd(),temp_path)
        result = scan_pdf(fullpath)
        result1 = mapping(fullpath)
        os.remove(temp_path)

        return jsonify({"result": result,
                        "resul1":result1})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/output/<filename>")
def serve_output_image(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)



@app.route("/process-images", methods=["GET"])
def process_images():
    try:
        if os.path.exists(OUTPUT_FOLDER):
            shutil.rmtree(OUTPUT_FOLDER)  # delete folder
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)  # recreate clean folder

        if not os.path.exists(IMAGES_FOLDER):
            return jsonify({"error": f"Folder {IMAGES_FOLDER} does not exist"}), 400

        image_files = []
        for root, _, files in os.walk(IMAGES_FOLDER):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_files.append(os.path.join(root, f))

        if not image_files:
            return jsonify({"message": "No images found in images_by_sector"}), 200

        grouped_detections = {
            "Sector_A": [],
            "Sector_B": [],
            "Sector_C": [],
            "Unsorted": []
        }


        for img_path in image_files:
            img_name = os.path.basename(img_path)
            
            sector = "Unsorted"  # default
            if "Sector_A" in img_path:
                sector = "Sector_A"
            elif "Sector_B" in img_path:
                sector = "Sector_B"
            elif "Sector_C" in img_path:
                sector = "Sector_C"


            # Run YOLO detection
            results = model.predict(source=img_path, save=False, verbose=False)

            for result in results:
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                counts = Counter(class_ids)
                summary = {class_names[c]: int(counts[c]) for c in counts}

                # Save annotated image
                img = result.plot()
                save_path = os.path.join(OUTPUT_FOLDER, img_name)
                cv2.imwrite(save_path, img)

                image_url = f"/output/{img_name}"  # you can serve this with another Flask route

                detection_data = {
                    "image": img_name,
                    "detections": summary,
                    "image_url": image_url
                }
                
                grouped_detections[sector].append(detection_data)

        del grouped_detections["Unsorted"]
        return jsonify({
            "message": "All images processed successfully",
            "total_images": len(image_files),
            "detections": grouped_detections
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/masking",methods=["GET"])
def extract():
    output_dir=os.path.join(os.getcwd(),"mapping","output")
    if not os.path.exists(output_dir):
        return jsonify({"error": "Output folder does not exist"}),404
    
    images = [f for f in os.listdir(output_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    
    image_urls = [f"/masking/{fname}" for fname in images]

    return jsonify({"images": image_urls})


@app.route("/masking/<filename>", methods=["GET"])
def get_masked_image(filename):
    output_dir = os.path.join(os.getcwd(), "mapping", "output")
    return send_from_directory(output_dir, filename)


@app.route("/")
def home():
    return "Hello, Flask!"



if __name__ == "__main__":
    app.run(debug=True)

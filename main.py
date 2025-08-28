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
from dotenv import load_dotenv
from test import mapping
# from datetime import datetime
import datetime
from utils.jwt_utils import generate_access_token
from middleware.check_auth import check_auth
from bson import ObjectId
# from access import main
from access3 import main

app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
db = get_db()
users_collection=db["users"]
files_collection = db["files"]



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
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024 




@app.route("/signup", methods=["POST"])
def signup():
    data = request.json
    name = data.get("name")
    email = data.get("email")
    password = data.get("password")
    role = data.get("role", "user")

    if not email or not password:
        return jsonify({"error": "email and password are required"}), 400

    if users_collection.find_one({"email": email}):
        return jsonify({"error": "User already exists"}), 400

    hashed_pw = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    users_collection.insert_one({
        "name":name,
        "email": email,
        "password": hashed_pw,
        "role": role
    })

    return jsonify({"message": "User created successfully"}), 201

@app.route("/login", methods=["POST"])
def login():
    data = request.json
    email = data.get("email")
    password = data.get("password")

    user = users_collection.find_one({"email": email})
    
    if not user:
        return jsonify({"error": "Invalid email or password"}), 401

    if bcrypt.checkpw(password.encode("utf-8"), user["password"].encode("utf-8")):
        user_id=str(user["_id"])
        payload = {
            "_id":user_id,
            "email": email,
            "role": user.get("role", "user")
        }
        token = generate_access_token(payload)

        return jsonify({
            "message": "Login successful",
            "email": email,
            "role": user.get("role", "user"),
            "access_token": token
        }), 200

    return jsonify({"error": "Invalid email or password"}), 401


@app.route("/users", methods=["GET"])
def users():
    try:
        users = list(users_collection.find(
            {"role": "user"},   
            {"_id": 0, "password": 0}  
        ))
        

        return jsonify({"success": True, "data": users}), 200
    except Exception as e:
        print("Error:", e)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/extract-page", methods=["POST"])
@check_auth("user")
def extract_page():
    user = request.user
    user_id = user.get("_id")

    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        uploaded_file = request.files["file"]

        if uploaded_file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        # ---- cleanup old images_by_sector before processing new pdf ----
        images_folder = os.path.join(os.getcwd(), "images_by_sector")
        if os.path.exists(images_folder):
            shutil.rmtree(images_folder)
        os.makedirs(images_folder, exist_ok=True)

        # ---- save uploaded PDF ----
        os.makedirs("uploads", exist_ok=True)
        temp_path = os.path.join("uploads", uploaded_file.filename)
        uploaded_file.save(temp_path)

        fullpath = os.path.join(os.getcwd(), temp_path)

        # ---- process PDF ----
        result = scan_pdf(fullpath)
        result1 = mapping(fullpath)
        # ---- save metadata to DB ----
        files_collection.insert_one({
            "filename": uploaded_file.filename,
            "user_id": ObjectId(user_id),
            "path": temp_path,
            "uploaded_at": datetime.datetime.utcnow()
        })
        result2 = main(fullpath)

        return jsonify({
            "filename": uploaded_file.filename,
            "user_id": user_id,
            "result": result,
            # "result1": result1
        })


    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/output/<path:filename>")
def serve_output_image(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)




@app.route("/process-images", methods=["GET"])
def process_images():
    try:
        # ✅ Find latest uploaded file (or modify if you want all)
        last_file = files_collection.find_one(sort=[("_id", -1)])
        if not last_file:
            return jsonify({"error": "No uploaded file found"}), 400

        # Unique output folder per file
        file_id = str(last_file["_id"])
        output_folder = os.path.join(OUTPUT_FOLDER, file_id)
        os.makedirs(output_folder, exist_ok=True)

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

            # Detect sector from path
            sector = "Unsorted"
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

                # ✅ Save annotated image in unique output/<file_id>/<sector>
                sector_folder = os.path.join(output_folder, sector)
                os.makedirs(sector_folder, exist_ok=True)
                save_path = os.path.join(sector_folder, img_name)

                img = result.plot()
                cv2.imwrite(save_path, img)

                # image_url includes file_id + sector
                image_url = f"/output/{file_id}/{sector}/{img_name}"

                detection_data = {
                    "image": img_name,
                    "detections": summary,
                    "image_url": image_url
                }

                grouped_detections[sector].append(detection_data)

        # Remove Unsorted if empty
        if not grouped_detections["Unsorted"]:
            del grouped_detections["Unsorted"]

        response_data = {
            "message": "All images processed successfully",
            "total_images": len(image_files),
            "detections": grouped_detections,
            "file_id": file_id
        }

        # ✅ Update the same file document with detections
        files_collection.update_one(
            {"_id": last_file["_id"]},
            {"$set": {"detections": grouped_detections, "processed_path": output_folder}}
        )

        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# @app.route("/access",methods=["GET"])
# def tracing():
#     try:
        
#     except Exception as e:
#         return jsonify({"message": str(e)}),500



@app.route("/masking", methods=["GET"])
def extract():
    try:
        # ✅ Find latest uploaded file
        last_file = files_collection.find_one(sort=[("_id", -1)])
        if not last_file:
            return jsonify({"error": "No uploaded file found"}), 400

        file_id = str(last_file["_id"])

        # ---- Run mapping() for masking ----
        pdf_path = last_file["path"]  # path of the uploaded PDF
        mapping_result = mapping(pdf_path)  # returns input_image, annotated_image, areas

        # ---- Prepare masking folder ----
        base_output = os.path.join("mapping", "output")
        masking_folder = os.path.join(base_output, file_id)
        os.makedirs(masking_folder, exist_ok=True)

        # Move/copy annotated image to file-specific folder
        annotated_filename = os.path.basename(mapping_result["annotated_image"])
        dest_path = os.path.join(masking_folder, annotated_filename)
        if not os.path.exists(dest_path):
            shutil.copy2(mapping_result["annotated_image"], dest_path)

        # Construct URL for frontend
        image_url = f"/masking/{file_id}/{annotated_filename}"

        # Update DB with masking info
        files_collection.update_one(
            {"_id": last_file["_id"]},
            {"$set": {
                "masking": [image_url],
                "masking_path": masking_folder,
                "masking_areas": mapping_result["areas_sqft"]
            }}
        )

        # Response
        response_data = {
            "file_id": file_id,
            "masking_image": image_url,
            "areas_sqft": mapping_result["areas_sqft"]
        }

        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route("/results/<filename>", methods=["GET"])
def get_results_image(filename):
    masking_folder = os.path.join(os.getcwd(),"results")
    
    return send_from_directory(masking_folder, filename)



@app.route("/masking/<file_id>/<filename>", methods=["GET"])
def get_masked_image(file_id, filename):
    masking_folder = os.path.join("mapping", "output", file_id)
    return send_from_directory(masking_folder, filename)


@app.route("/")
def home():
    return "Hello, Flask!"

@app.route("/categorize-pdfs", methods=["GET"])
def categorize_pdfs():
    try:
        # Make sure uploads folder exists
        if not os.path.exists(UPLOAD_FOLDER):
            return jsonify({"error": f"Folder {UPLOAD_FOLDER} does not exist"}), 400

        pdf_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.lower().endswith(".pdf")]

        if not pdf_files:
            return jsonify({"message": "No PDFs found in uploads"}), 200

        # Create new collection for categorized PDFs
        categorized_collection = db["categorized_pdfs"]

        categorized_collection.delete_many({})  # optional: clear old data

        for pdf in pdf_files:
            pdf_path = os.path.join(UPLOAD_FOLDER, pdf)

            # Detect sector based on filename (you can adjust the logic)
            sector = "Unsorted"
            if "Sector_A" in pdf:
                sector = "Sector_A"
            elif "Sector_B" in pdf:
                sector = "Sector_B"
            elif "Sector_C" in pdf:
                sector = "Sector_C"

            # Insert into new collection
            categorized_collection.insert_one({
                "filename": pdf,
                "path": pdf_path,
                "sector": sector,
                "uploaded_at": datetime.utcnow()
            })

        return jsonify({
            "message": "PDFs categorized successfully",
            "total_pdfs": len(pdf_files)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/analyze',methods=["GET"])
def analyzdme():
    try:
        query=[
    {
        '$addFields': {
            '_id': {
                '$toString': '$_id'
            }, 
            'user_id': {
                '$toString': '$user_id'
            }, 
            'sectorA_caution': {
                '$sum': {
                    '$map': {
                        'input': '$detections.Sector_A', 
                        'as': 's', 
                        'in': {
                            '$ifNull': [
                                '$$s.detections.caution', 0
                            ]
                        }
                    }
                }
            }, 
            'sectorB_caution': {
                '$sum': {
                    '$map': {
                        'input': '$detections.Sector_B', 
                        'as': 's', 
                        'in': {
                            '$ifNull': [
                                '$$s.detections.caution', 0
                            ]
                        }
                    }
                }
            }, 
            'sectorC_caution': {
                '$sum': {
                    '$map': {
                        'input': '$detections.Sector_C', 
                        'as': 's', 
                        'in': {
                            '$ifNull': [
                                '$$s.detections.caution', 0
                            ]
                        }
                    }
                }
            }
        }
    }
]
    

        result=list(db["files"].aggregate(query))
        return jsonify({"message":"fetched data successfully","data":result}), 200
    except Exception as e:
        return str(e),None

@app.route("/dashboard",methods=["GET"])
def dashboard():
    try:
        file_id=request.args.get("_id")
        
        query1=[
    {
        '$match': {
            '_id': ObjectId(file_id)
        }
    }, {
        '$addFields': {
            '_id': {
                '$toString': '$_id'
            }
        }
    }, {
        '$project': {
            'masking_areas': 1
        }
    }
]
        
        
        query=[
            {
        '$match': {
            '_id': ObjectId(file_id)
        }
    },
    {
        '$project': {
            'sectorA_caution': {
                '$sum': {
                    '$map': {
                        'input': '$detections.Sector_A', 
                        'as': 's', 
                        'in': {
                            '$ifNull': [
                                '$$s.detections.caution', 0
                            ]
                        }
                    }
                }
            }, 
            'sectorA_warning': {
                '$sum': {
                    '$map': {
                        'input': '$detections.Sector_A', 
                        'as': 's', 
                        'in': {
                            '$ifNull': [
                                '$$s.detections.warning', 0
                            ]
                        }
                    }
                }
            }, 
            'sectorA_notice': {
                '$sum': {
                    '$map': {
                        'input': '$detections.Sector_A', 
                        'as': 's', 
                        'in': {
                            '$ifNull': [
                                '$$s.detections.notice', 0
                            ]
                        }
                    }
                }
            }, 
            'sectorB_caution': {
                '$sum': {
                    '$map': {
                        'input': '$detections.Sector_B', 
                        'as': 's', 
                        'in': {
                            '$ifNull': [
                                '$$s.detections.caution', 0
                            ]
                        }
                    }
                }
            }, 
            'sectorB_warning': {
                '$sum': {
                    '$map': {
                        'input': '$detections.Sector_B', 
                        'as': 's', 
                        'in': {
                            '$ifNull': [
                                '$$s.detections.warning', 0
                            ]
                        }
                    }
                }
            }, 
            'sectorB_notice': {
                '$sum': {
                    '$map': {
                        'input': '$detections.Sector_B', 
                        'as': 's', 
                        'in': {
                            '$ifNull': [
                                '$$s.detections.notice', 0
                            ]
                        }
                    }
                }
            }, 
            'sectorC_caution': {
                '$sum': {
                    '$map': {
                        'input': '$detections.Sector_C', 
                        'as': 's', 
                        'in': {
                            '$ifNull': [
                                '$$s.detections.caution', 0
                            ]
                        }
                    }
                }
            }, 
            'sectorC_warning': {
                '$sum': {
                    '$map': {
                        'input': '$detections.Sector_C', 
                        'as': 's', 
                        'in': {
                            '$ifNull': [
                                '$$s.detections.warning', 0
                            ]
                        }
                    }
                }
            }, 
            'sectorC_notice': {
                '$sum': {
                    '$map': {
                        'input': '$detections.Sector_C', 
                        'as': 's', 
                        'in': {
                            '$ifNull': [
                                '$$s.detections.notice', 0
                            ]
                        }
                    }
                }
            }
        }
    }, {
        '$group': {
            '_id': None, 
            'sectorA_caution': {
                '$sum': '$sectorA_caution'
            }, 
            'sectorA_warning': {
                '$sum': '$sectorA_warning'
            }, 
            'sectorA_notice': {
                '$sum': '$sectorA_notice'
            }, 
            'sectorB_caution': {
                '$sum': '$sectorB_caution'
            }, 
            'sectorB_warning': {
                '$sum': '$sectorB_warning'
            }, 
            'sectorB_notice': {
                '$sum': '$sectorB_notice'
            }, 
            'sectorC_caution': {
                '$sum': '$sectorC_caution'
            }, 
            'sectorC_warning': {
                '$sum': '$sectorC_warning'
            }, 
            'sectorC_notice': {
                '$sum': '$sectorC_notice'
            }, 
            'total_caution': {
                '$sum': {
                    '$add': [
                        '$sectorA_caution', '$sectorB_caution', '$sectorC_caution'
                    ]
                }
            }, 
            'total_warning': {
                '$sum': {
                    '$add': [
                        '$sectorA_warning', '$sectorB_warning', '$sectorC_warning'
                    ]
                }
            }, 
            'total_notice': {
                '$sum': {
                    '$add': [
                        '$sectorA_notice', '$sectorB_notice', '$sectorC_notice'
                    ]
                }
            }
        }
    }
    
    
]  
        
        result=list(db["files"].aggregate(query))
        result1=list(db["files"].aggregate(query1))
        final_data={
            "final_result":result,
            "final_result2":result1
        }
        return jsonify({"message":"fetch data succeffulye","data":final_data}),200
    except Exception as e:
        return str(e),None
    
    
@app.route("/files", methods=["GET"])
def get_filenames():
    pipeline = [
    {
        '$project': {
            'filename': 1
        }
    }, {
        '$addFields': {
            '_id': {
                '$toString': '$_id'
            }
        }
    }
]
    result = list(files_collection.aggregate(pipeline))
   
    
   
    return jsonify({"message":"fetched data successfully","data":result}), 200



if __name__ == "__main__":
    app.run(debug=True)

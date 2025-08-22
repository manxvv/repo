from ultralytics import YOLO
import os
import shutil
import pandas as pd

# ======================
# CONFIG
# ======================
MODEL_PATH = r"C:\Users\lenovo\Desktop\caution2k-20250818T051503Z-1-001\best.pt"
SOURCE_DIR = r"C:\Users\lenovo\Desktop\caution2k-20250818T051503Z-1-001\images_by_sector"
OUTPUT_DIR = r"C:\Users\lenovo\Desktop\caution2k-20250818T051503Z-1-001\predictions"

# ======================
# LOAD MODEL
# ======================
model = YOLO(MODEL_PATH)

# ======================
# CLEAN OLD RESULTS
# ======================
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

# ======================
# COUNT STORAGE
# ======================
sector_counts = {}

# ======================
# RUN DETECTION ON EACH SUBFOLDER
# ======================
print("🚀 Running detection...")
for subdir in os.listdir(SOURCE_DIR):
    sub_path = os.path.join(SOURCE_DIR, subdir)
    if os.path.isdir(sub_path) and subdir.lower() != "unsorted":  # skip 'Unsorted'
        results = model.predict(
            source=sub_path,
            save=True,
            project=OUTPUT_DIR,
            name=subdir,  # each sector gets its own result folder
            verbose=False
        )
        
        # Count "caution" detections (assuming class name is 'caution')
        count = 0
        for r in results:
            boxes = r.boxes
            names = model.names
            for cls_id in boxes.cls.cpu().numpy():
                if names[int(cls_id)].lower() == "caution":
                    count += 1

        sector_counts[subdir] = count

print(f"✅ Done! Predictions saved in: {OUTPUT_DIR}")
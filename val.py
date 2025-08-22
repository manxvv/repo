import random
from pathlib import Path

# Path to your dataset folder
dataset_dir = Path(r"C:\Users\Admin\OneDrive\Documents\Desktop\caution2k")  # <-- change to your folder

train_file = dataset_dir / "train.txt"
valid_file = dataset_dir / "valid.txt"

# Read all image paths
with open(train_file, "r") as f:
    images = f.read().strip().split("\n")

# Shuffle so split is random
random.shuffle(images)

# 80% train, 20% valid
split_idx = int(0.8 * len(images))
train_list = images[:split_idx]
valid_list = images[split_idx:]

# Save new files
with open(train_file, "w") as f:
    f.write("\n".join(train_list))

with open(valid_file, "w") as f:
    f.write("\n".join(valid_list))

print(f"✅ Train images: {len(train_list)}, Valid images: {len(valid_list)}")
print(f"📂 Saved: {train_file} and {valid_file}")

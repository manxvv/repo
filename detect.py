from ultralytics import YOLO
import cv2
from collections import Counter 

# Load the trained model
model = YOLO("best.pt")

class_names = ["caution", "warning", "notice"]

# Run prediction (don't use show=True here, we will handle display manually)
results = model.predict(source=r"C:\Users\Admin\OneDrive\Documents\Desktop\caution2k-20250818T051503Z-1-001\images_raw\40.6.jpg", save=True)

# Loop through results and display each image
for result in results:
    # Convert the result to an OpenCV image
    img = result.plot()  # Draw bounding boxes on the image
    
    # Count predictions
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    counts = Counter(class_ids)

    # Print the summary in terminal
    summary = "  ".join([f"{class_names[c]}-{counts[c]}" for c in counts])
    print(summary)  

    # # Count predictions
    # class_ids = result.boxes.cls.cpu().numpy().astype(int)  # class indices
    # counts = Counter(class_ids)

    # # Build the summary text
    # summary = "  ".join([f"{class_names[c]}-{counts[c]}" for c in counts])

    # # Overlay text on the image
    # cv2.putText(img, summary, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
    #             1, (0, 255, 0), 2, cv2.LINE_AA)

    img = cv2.resize(img, (1000, 400))

    # Show the image until 'q' is pressed
    while True:
        cv2.imshow("Detection", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()

# === PDF input → extract image under 2nd heading → run YOUR masking code (no CSV) ===
import fitz  # PyMuPDF
import cv2
import numpy as np
import os

# --------- PDF as input + extract image under 2nd heading into \input ---------
# pdf_path   = r"C:\Users\lenovo\Downloads\signal.pdf"



def mapping(PDF_PATH):
    heading    = "Predictive Cumulative MPE Contribution from All Sources: 3D Top View"
    base_dir   = os.path.join(os.getcwd(), "mapping")
    os.makedirs(base_dir, exist_ok=True)
    input_dir  = os.path.join(base_dir, "input")   # folder 1: input image
    output_dir = os.path.join(base_dir, "output")  # folder 2: annotated output
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    doc = fitz.open(PDF_PATH)
    occurrence = 0
    extracted_image_path = None

    for pno in range(len(doc)):
        page = doc[pno]
        rects = page.search_for(heading.strip())
        if not rects:
            continue
        for r in sorted(rects, key=lambda x: (x.y0, x.x0)):
            occurrence += 1
            if occurrence == 2:
                heading_bottom_y = r.y1 + 5  # small margin below heading
                best = None  # (area, xref)
                for img in page.get_images(full=True):
                    xref = img[0]
                    try:
                        placements = page.get_image_rects(xref)
                    except Exception:
                        placements = []
                    for pr in placements:
                        if pr.y0 >= heading_bottom_y:
                            area = pr.width * pr.height
                            if best is None or area > best[0]:
                                best = (area, xref)
                if best is None:
                    raise RuntimeError("No raster image found under the 2nd heading.")
                info = doc.extract_image(best[1])
                ext  = info.get("ext", "png")
                extracted_image_path = os.path.join(input_dir, f"extracted_under_heading.{ext}")
                with open(extracted_image_path, "wb") as f:
                    f.write(info["image"])
                break
        if extracted_image_path:
            break

    if not extracted_image_path:
        raise RuntimeError("Heading not found twice in the document.")

    # --------- From here on: YOUR MASKING CODE (unchanged logic) ---------

    # === Step 1: Load the image from local path ===
    image_path = extracted_image_path  # use the extracted image as input
    image = cv2.imread(image_path)

    # Check if image loaded
    if image is None:
        raise FileNotFoundError("❌ Failed to load image. Please check the file path.")

    # === Step 2: Copy for output ===
    output = image.copy()

    # === Step 4: Constants for area calculation ===
    grid_size_px = 82     # calibrated for PDF-extracted image
    grid_area_ft2 = 100
    pixel_to_ft2 = grid_area_ft2 / (grid_size_px ** 2)

    # scale contour-area thresholds with resolution
    SCALE = (grid_size_px / 50.0) ** 2
    MIN_BG_AREA  = int(100 * SCALE)          # for Blue/Green (was 100)
    MIN_RED_AREA = max(1, int(3 * SCALE))    # for Red small blobs (was 3)
    MAX_RED_AREA = int(300 * SCALE)          # for Red upper bound (was 300)

    # === Step 5: Convert to HSV ===
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # === Step 6: Define color ranges ===
    blue_lower = np.array([90, 50, 50])
    blue_upper = np.array([130, 255, 255])

    green_lower = np.array([35, 50, 50])
    green_upper = np.array([85, 255, 255])

    red_lower1 = np.array([0, 50, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 50, 50])
    red_upper2 = np.array([180, 255, 255])
    red_lower3 = np.array([10, 50, 50])
    red_upper3 = np.array([25, 255, 255])

    kernel = np.ones((3, 3), np.uint8)

    # === Step 7: Create original masks ===
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)

    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

    red1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red3 = cv2.inRange(hsv, red_lower3, red_upper3)
    red_mask = cv2.bitwise_or(cv2.bitwise_or(red1, red2), red3)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

    # === Step 8 (REVISED): Build a *clean* balloon-only mask ===
    combined_mask = cv2.bitwise_or(blue_mask, green_mask)
    combined_mask = cv2.bitwise_or(combined_mask, red_mask)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

    open_k = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, open_k)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(combined_mask, connectivity=8)
    balloon_mask = np.zeros_like(combined_mask)

    MIN_COMP_AREA = 3000
    MIN_GREEN_OVERLAP_PX = 150

    for i in range(1, num):  # 0 is background
        area = stats[i, cv2.CC_STAT_AREA]
        if area < MIN_COMP_AREA:
            continue
        comp = (labels == i).astype(np.uint8) * 255
        overlap = cv2.countNonZero(cv2.bitwise_and(comp, green_mask))
        if overlap >= MIN_GREEN_OVERLAP_PX:
            balloon_mask = cv2.bitwise_or(balloon_mask, comp)

    close_k = np.ones((7, 7), np.uint8)
    balloon_mask = cv2.morphologyEx(balloon_mask, cv2.MORPH_CLOSE, close_k)

    # === Step 9: Mask the image to white background ===
    masked_output = np.full_like(output, 255)  # pure white background
    masked_output[balloon_mask == 255] = output[balloon_mask == 255]

    # === Limit detection masks to balloon area ===
    blue_mask = cv2.bitwise_and(blue_mask, balloon_mask)
    green_mask = cv2.bitwise_and(green_mask, balloon_mask)
    red_mask = cv2.bitwise_and(red_mask, balloon_mask)

    # === Step 11: Bounding Boxes and Area Calculation ===
    color_info = {
        "Blue": (blue_mask, (255, 0, 0)),
        "Green": (green_mask, (0, 255, 0)),
        "Red": (red_mask, (0, 0, 255)),
    }

    annotated_output = masked_output.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    total_areas = {}

    for color_name, (mask, box_color) in color_info.items():
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_pixels = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)

            # scaled thresholds
            if color_name == "Red":
                if area < MIN_RED_AREA or area > MAX_RED_AREA:
                    continue
            else:
                if area < MIN_BG_AREA:
                    continue

            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(annotated_output, (x, y), (x + w, y + h), box_color, 2)
            area_ft2 = area * pixel_to_ft2
            total_pixels += area

            cv2.putText(
                annotated_output,
                f"{color_name}: {area_ft2:.1f} sqft",
                (x, max(15, y - 5)),
                font, 0.5, box_color, 2, cv2.LINE_AA
            )
        total_areas[color_name] = total_pixels * pixel_to_ft2

    # === Totals on top ===
    cv2.putText(annotated_output, f"Total Blue Area: {total_areas.get('Blue', 0.0):.1f} sqft",
                (20, 30), font, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(annotated_output, f"Total Green Area: {total_areas.get('Green', 0.0):.1f} sqft",
                (20, 60), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(annotated_output, f"Total Red Area: {total_areas.get('Red', 0.0):.1f} sqft",
                (20, 90), font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    # === Save output (annotated only) ===
    base = os.path.splitext(os.path.basename(image_path))[0]
    annotated_path = os.path.join(output_dir, f"{base}.jpeg")  # same name as input
    cv2.imwrite(annotated_path, annotated_output)
    
   

    print(f"✅ Saved:\n  Input image:  {extracted_image_path}\n  Output image: {annotated_path}\n")
    print({
    "Blue": round(total_areas.get("Blue", 0.0), 1),
    "Green": round(total_areas.get("Green", 0.0), 1),
    "Red": round(total_areas.get("Red", 0.0), 1)
})
    result = {
        "input_image": extracted_image_path,
        "annotated_image": annotated_path,
        "areas_sqft": {
            "Blue": round(total_areas.get("Blue", 0.0), 1),
            "Green": round(total_areas.get("Green", 0.0), 1),
            "Red": round(total_areas.get("Red", 0.0), 1)
        }
    }
    return result
   
    

    # annotated_path = os.path.join(output_dir, f"{base}_annotated.png")
    # cv2.imwrite(annotated_path, annotated_output)


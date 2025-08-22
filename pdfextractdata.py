import fitz  # PyMuPDF
import os
import re
from typing import List, Tuple

# =========================
# CONFIG
# =========================
# PDF_PATH = r"C:\Users\lenovo\Desktop\caution2k-20250818T051503Z-1-001\uploads\10087226_36504_CLL05824_CLL25824_RFSSR_04-22-2025.pdf"
OUTPUT_DIR = "images_by_sector"

# Create subfolders
SECTOR_DIRS = {
    "Sector_A": os.path.join(OUTPUT_DIR, "Sector_A"),
    "Sector_B": os.path.join(OUTPUT_DIR, "Sector_B"),
    "Sector_C": os.path.join(OUTPUT_DIR, "Sector_C"),
    "Unsorted": os.path.join(OUTPUT_DIR, "Unsorted"),
}


FORCE_UNSORTED = {
    "at&t sector a antennas (a4, a3, a2, a1)",
    "at&t sector b antennas (a8, a7, a6, a5)",
    "at&t sector c antennas (a12, a11, a10, a9)",
}



# Start/stop markers
START_MARK = re.compile(r"Site Photograph", re.I)
STOP_MARK  = re.compile(r"Emission Measurements and Predictions", re.I)

# Geometry
ROW_TOL         = 28.0   # how close images top y must be to be same row
CAPTION_Y_LIMIT = 120.0  # how far below row we search for caption
LINE_TOL        = 2.5    # tolerance for joining words into same line
GAP_FACTOR      = 0.6    # fraction of word height → if x-gap > this, treat as "big space"

def norm_caption(s: str) -> str:
    # collapse whitespace and lowercase
    return re.sub(r"\s+", " ", s).strip().lower()

# =========================
# HELPERS
# =========================
def save_pix_as_png(pix: fitz.Pixmap, out_path: str) -> None:
    pix.save(out_path)


def words_to_lines(words: List[Tuple]) -> List[dict]:
    """
    Group words into lines.
    Preserve wide gaps as multiple spaces so captions can be split correctly.
    """
    if not words:
        return []

    words_sorted = sorted(words, key=lambda w: (w[1], w[0]))
    lines, current_line, current_y = [], [], None

    for w in words_sorted:
        x0, y0, x1, y1, text, *_ = w
        if current_y is None:
            current_y, current_line = y0, [w]
            continue

        if abs(y0 - current_y) <= LINE_TOL:
            current_line.append(w)
        else:
            # flush
            lines.append(build_line(current_line))
            current_y, current_line = y0, [w]

    if current_line:
        lines.append(build_line(current_line))

    return sorted(lines, key=lambda L: (L["y0"], L["x0"]))


def build_line(words: List[Tuple]) -> dict:
    """Turn word list into a single line dict, preserving wide gaps."""
    words = sorted(words, key=lambda w: w[0])
    parts = [words[0][4]]
    for prev, cur in zip(words, words[1:]):
        gap = cur[0] - prev[2]
        avg_height = (prev[3] - prev[1] + cur[3] - cur[1]) / 2
        if gap > avg_height * GAP_FACTOR:
            parts.append("  ")  # insert extra spaces for wide gap
        else:
            parts.append(" ")
        parts.append(cur[4])

    lx0 = min(t[0] for t in words)
    ly0 = min(t[1] for t in words)
    lx1 = max(t[2] for t in words)
    ly1 = max(t[3] for t in words)
    txt = "".join(parts).strip()

    return {"x0": lx0, "y0": ly0, "x1": lx1, "y1": ly1, "text": txt}


def order_images_into_rows(images: List[dict]) -> List[List[dict]]:
    """Group images into rows by y0, then sort each row left→right."""
    if not images:
        return []
    images = sorted(images, key=lambda o: o["bbox"].y0)
    rows = []
    for oc in images:
        placed = False
        for row in rows:
            ref_y = row[0]["bbox"].y0
            if abs(oc["bbox"].y0 - ref_y) <= ROW_TOL:
                row.append(oc)
                placed = True
                break
        if not placed:
            rows.append([oc])
    for row in rows:
        row.sort(key=lambda o: o["bbox"].x0)
    return rows


def find_row_caption(row: List[dict], lines: List[dict]) -> List[str]:
    """Find the line immediately under the row and split among images."""
    if not row:
        return []

    bottom_y = max(oc["bbox"].y1 for oc in row)
  
  
    # candidate lines below row
    candidates = [
        L for L in lines
        if L["y0"] > bottom_y and (L["y0"] - bottom_y) <= CAPTION_Y_LIMIT
    ]
    if not candidates:
        return ["No caption"] * len(row)

    # pick nearest below
    candidates.sort(key=lambda L: L["y0"] - bottom_y)
    caption_line = candidates[0]["text"]

    # split caption by 2+ spaces
    parts = re.split(r"\s{2,}", caption_line)
    if len(parts) < len(row):
        # fallback: equal split by word count
        words = caption_line.split()
        k = len(words) // len(row) if len(row) else 0
        parts = [" ".join(words[i*k:(i+1)*k]) for i in range(len(row))]
        if len(parts) < len(row):
            parts += [""] * (len(row) - len(parts))

    return [p.strip() if p.strip() else "No caption" for p in parts[:len(row)]]


def classify_caption(caption: str) -> str:
    """Return folder name based on caption text."""
    # === NEW (Option A): force certain captions to Unsorted ===
    n = norm_caption(caption)
    if n in FORCE_UNSORTED:
        return "Unsorted"
    
    cap_low = n
    if "sector a" in cap_low:
        return "Sector_A"
    elif "sector b" in cap_low:
        return "Sector_B"
    elif "sector c" in cap_low:
        return "Sector_C"
    else:
        return "Unsorted"


# =========================
# MAIN
# =========================
def scan_pdf(PDF_PATH):
    for d in SECTOR_DIRS.values():
        os.makedirs(d, exist_ok=True)
    doc = fitz.open(PDF_PATH)
    start_count, stop_count, extracting, total_saved = 0, 0, False, 0

    for pno, page in enumerate(doc, start=1):
        full_text = page.get_text("text")
        start_count += len(START_MARK.findall(full_text))
        stop_count  += len(STOP_MARK.findall(full_text))

        if (not extracting) and start_count >= 2:
            extracting = True
            print(f"Started extraction at page {pno}")

        if extracting and stop_count < 2:
            page_images = page.get_images(full=True)
            occurrences = []
            for im in page_images:
                bbox = page.get_image_bbox(im)
                if (bbox.width < 40) or (bbox.height < 40):
                    continue
                occurrences.append({"img": im, "bbox": bbox})

            rows = order_images_into_rows(occurrences)
            lines = words_to_lines(page.get_text("words"))

            img_idx = 0
            for row in rows:
                captions = find_row_caption(row, lines)
                for i, oc in enumerate(row):
                    img_idx += 1
                    caption = captions[i]
                    sector = classify_caption(caption)

                    # Save image into correct folder
                    xref = oc["img"][0]
                    pix = fitz.Pixmap(doc, xref)
                    out_name = f"page{pno}_img{img_idx}.png"
                    out_path = os.path.join(SECTOR_DIRS[sector], out_name)
                    save_pix_as_png(pix, out_path)

                    print(f"Page {pno}, Image {img_idx} → Caption: {caption}  → {sector}")
                    total_saved += 1

        if extracting and stop_count >= 2:
            print(f"Stopped extraction at page {pno}")
            break

    doc.close()
    print(f"Saved {total_saved} images into '{OUTPUT_DIR}' (sorted by sector)")


# if __name__ == "__main__":
#     main()

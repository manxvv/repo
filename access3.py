# access_pdf_to_detection.py
import os, re, sys, math
import cv2
import fitz  # PyMuPDF
import numpy as np
from ultralytics import YOLO
import easyocr
from db import get_db

db = get_db()
users_collection=db["users"]
files_collection = db["files"]


# ---------- OCR (init once) ----------
OCR = easyocr.Reader(['en'], gpu=False)

# ===================== OCR HELPERS =====================

def ocr_boxes(img, min_conf=0.45):
    """Return [(text, conf, (x1,y1,x2,y2))] from EasyOCR on BGR/RGB image."""
    res = OCR.readtext(img)
    out = []
    for poly, text, conf in res:
        if float(conf) < min_conf:
            continue
        xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
        x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        out.append((text.strip(), float(conf), (x1, y1, x2, y2)))
    return out


def find_access_point(img):
    """
    Robust ACCESS finder: multi-scale + contrast enhance + fuzzy matching.
    Returns {'center':(cx,cy), 'box':(x1,y1,x2,y2)} or None.
    """
    from difflib import SequenceMatcher

    def _norm(t): return re.sub(r'[^A-Z0-9]', '', (t or '').upper())
    def _sim(t):  return SequenceMatcher(None, _norm(t), "ACCESS").ratio()

    def enhance(bgr):
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        L = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(L)
        return cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2BGR)

    scales = [1.6, 2.0, 2.4]
    cands = []
    print(cands,"SDSDsdsds")
    for S in scales:
        up = cv2.resize(img, None, fx=S, fy=S, interpolation=cv2.INTER_LINEAR)
        up = enhance(up)
        for poly, text, conf in OCR.readtext(up):
            score = _sim(text)
            if score < 0.72 and _norm(text) != "ACCESS":
                continue
            xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
            x1,y1,x2,y2 = min(xs),min(ys),max(xs),max(ys)
            # back to original coords
            x1o,y1o,x2o,y2o = int(x1/S),int(y1/S),int(x2/S),int(y2/S)
            cx, cy = (x1o+x2o)//2, (y1o+y2o)//2
            cands.append({
                "center": (cx,cy), "box": (x1o,y1o,x2o,y2o),
                "conf": float(conf), "score": float(score),
                "exact": _norm(text)=="ACCESS", "x": cx
            })
        if cands: break
    print(cands,"SDSDsdsds")

    if not cands:
        return None
    cands.sort(key=lambda c: (0 if c["exact"] else 1, c["x"], -c["score"], -c["conf"]))
    best = cands[0]
    return {"center": best["center"], "box": best["box"]}

# ===================== CAUTION DETECTION =====================

_YOLO_MODEL = None
_YOLO_PATH  = None

def detect_caution_yolo(img_bgr, weights_path, conf=0.28, imgsz=None, scales=(1.0, 1.6, 2.2)):
    """
    Multi-scale YOLO to catch small cautions. Returns [x1,y1,x2,y2,score] (orig coords).
    """
    global _YOLO_MODEL, _YOLO_PATH
    if (_YOLO_MODEL is None) or (_YOLO_PATH != weights_path):
        _YOLO_MODEL = YOLO(weights_path)
        _YOLO_PATH  = weights_path

    H, W = img_bgr.shape[:2]
    if imgsz is None:
        long_side = max(H, W)
        imgsz = int(np.clip(((long_side + 31)//32)*32, 960, 1536))

    boxes = []
    for S in scales:
        src = img_bgr if abs(S-1.0) < 1e-3 else cv2.resize(img_bgr, None, fx=S, fy=S)
        res = _YOLO_MODEL.predict(source=src, imgsz=imgsz, conf=conf, agnostic_nms=True, verbose=False)[0]
        if res.boxes is None:
            continue
        for b in res.boxes:
            x1,y1,x2,y2 = map(int, b.xyxy[0].cpu().numpy().tolist())
            score = float(b.conf[0].cpu().numpy())
            if abs(S-1.0) >= 1e-3:
                x1 = int(x1 / S); y1 = int(y1 / S); x2 = int(x2 / S); y2 = int(y2 / S)
            x1 = max(0, min(W-1, x1)); x2 = max(0, min(W-1, x2))
            y1 = max(0, min(H-1, y1)); y2 = max(0, min(H-1, y2))
            boxes.append([x1,y1,x2,y2,score])
    return nms_merge(boxes, iou_thr=0.50)


def detect_caution_ocr(img_bgr, min_conf=0.35, scales=(1.4, 1.8, 2.2)):
    """OCR fallback to find the 'CAUTION' header on labels. Returns [x1,y1,x2,y2,score]."""
    H, W = img_bgr.shape[:2]

    def enhance(bgr):
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        L = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(L)
        return cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2BGR)

    out = []
    for S in scales:
        up = cv2.resize(img_bgr, None, fx=S, fy=S, interpolation=cv2.INTER_LINEAR)
        up = enhance(up)
        for poly, text, conf in OCR.readtext(up):
            if float(conf) < min_conf:
                continue
            if not re.search(r'\bCAUTION\b', (text or ''), flags=re.I):
                continue
            xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
            x1,y1,x2,y2 = int(min(xs)),int(min(ys)),int(max(xs)),int(max(ys))
            if abs(S-1.0) >= 1e-3:
                x1 = int(x1 / S); y1 = int(y1 / S); x2 = int(x2 / S); y2 = int(y2 / S)
            x1 = max(0, min(W-1, x1)); x2 = max(0, min(W-1, x2))
            y1 = max(0, min(H-1, y1)); y2 = max(0, min(H-1, y2))
            out.append([x1,y1,x2,y2,float(conf)])
    return nms_merge(out, iou_thr=0.40)

# ---------- merge utilities ----------

def iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    iw = max(0, x2 - x1 + 1); ih = max(0, y2 - y1 + 1)
    inter = iw * ih
    a1 = (b1[2]-b1[0]+1) * (b1[3]-b1[1]+1)
    a2 = (b2[2]-b2[0]+1) * (b2[3]-b2[1]+1)
    denom = a1 + a2 - inter
    return (inter / denom) if denom > 0 else 0.0

def nms_merge(boxes, iou_thr=0.5):
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    kept = []
    for b in boxes:
        if all(iou(b, k) < iou_thr for k in kept):
            kept.append(b)
    return kept

def merge_yolo_ocr(yolo_boxes, ocr_boxes, iou_thr=0.25):
    """Union of YOLO and OCR boxes, replacing overlaps by higher score."""
    out = list(yolo_boxes)
    for b in ocr_boxes:
        placed = False
        for i, a in enumerate(out):
            if iou(a, b) >= iou_thr:
                if b[4] > a[4]:
                    out[i] = b
                placed = True
                break
        if not placed:
            out.append(b)
    return out

# ===================== ORDERING (VISIT ALL) =====================

def order_boxes_nearest(boxes, origin_xy):
    """
    Greedy nearest-neighbor tour starting at origin.
    Returns boxes in visiting order (covers ALL boxes).
    """
    if not boxes:
        return []
    rem = boxes[:]
    ordered = []
    curx, cury = origin_xy
    while rem:
        j = min(
            range(len(rem)),
            key=lambda k: (( (rem[k][0]+rem[k][2]) * 0.5 - curx )**2) +
                          (( (rem[k][1]+rem[k][3]) * 0.5 - cury )**2)
        )
        b = rem.pop(j)
        ordered.append(b)
        x1,y1,x2,y2,_ = b
        curx, cury = ( (x1+x2)*0.5, (y1+y2)*0.5 )
    return ordered

# ===================== DRAWING =====================

def _box_center(b):
    x1,y1,x2,y2,_ = b
    return int((x1+x2)//2), int((y1+y2)//2)

def draw_debug_with_path(image_bgr, access_pt, ordered_boxes, show_indices=True, path_thickness=2):
    img = image_bgr.copy()
    # ACCESS
    x1,y1,x2,y2 = access_pt['box']
    ax, ay = access_pt['center']
    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,255), 2)
    cv2.circle(img, (ax,ay), 6, (0,255,255), -1)
    cv2.putText(img, 'ACCESS', (x1, max(0,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    # Cautions + path
    pts = [(ax, ay)]
    for idx, b in enumerate(ordered_boxes, start=1):
        x1,y1,x2,y2,score = b
        cx, cy = _box_center(b)
        pts.append((cx, cy))
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,165,255), 2)
        if show_indices:
            cv2.putText(img, f"{idx}", (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,165,255), 2)

    if len(pts) >= 2:
        cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, (0,255,0), path_thickness)
        for i in range(len(pts)-1):
            cv2.arrowedLine(img, pts[i], pts[i+1], (0,255,0), path_thickness, tipLength=0.10)
    return img

# ===================== PDF → IMAGE (2nd "Site Scale Map") =====================

def extract_second_site_scale_map(pdf_path, out_img_path, keyword="Site Scale Map", margin_px=20, zoom=2.0):
    """
    Find the 2nd occurrence of `keyword`. Prefer the largest image block below it.
    Fallback: render region below heading to page bottom.
    """
    if not os.path.exists(pdf_path):
        return False
    os.makedirs(os.path.dirname(out_img_path), exist_ok=True)
    doc = fitz.open(pdf_path)

    # A) text search
    seen = set(); hits = []
    for pno in range(len(doc)):
        page = doc[pno]
        for q in (keyword, keyword.upper(), keyword.lower()):
            try:
                rects = page.search_for(q) or []
            except Exception:
                rects = []
            for r in rects:
                key = (pno, int(round(r.y1*1000)))
                if key not in seen:
                    seen.add(key); hits.append((pno, r.y1))
    # B) OCR top 35% if needed
    if len(hits) < 2:
        for pno in range(len(doc)):
            page = doc[pno]; pr = page.rect
            clip = fitz.Rect(pr.x0, pr.y0, pr.x1, pr.y0 + 0.35*pr.height)
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
            rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if rgb.shape[2] > 3: rgb = rgb[:, :, :3]
            for poly, txt, conf in OCR.readtext(rgb):
                if float(conf) < 0.45: continue
                if "site scale map" in (txt or '').lower():
                    y_px = max(pt[1] for pt in poly) + margin_px
                    y_page = (y_px / zoom) + clip.y0
                    key = (pno, int(round(y_page*1000)))
                    if key not in seen:
                        seen.add(key); hits.append((pno, y_page))
                        if len(hits) >= 2: break
            if len(hits) >= 2: break

    if len(hits) < 2:
        doc.close(); return False

    hits.sort(key=lambda t: (t[0], t[1]))
    pno, y_bottom = hits[1]
    page = doc[pno]; pr = page.rect
    raw = page.get_text("rawdict")

    # Try largest image block below the heading
    best_rect, best_area = None, 0.0
    page_area = max(1e-6, pr.width*pr.height)
    for block in raw.get("blocks", []):
        if block.get("type") != 1:  # image blocks only
            continue
        x0,y0,x1,y1 = block.get("bbox", (0,0,0,0))
        if y0 <= y_bottom:  # must be below heading
            continue
        area = max(0.0, (x1-x0)*(y1-y0))
        if area/page_area < 0.05:
            continue
        if area > best_area:
            best_area = area
            best_rect = fitz.Rect(x0,y0,x1,y1)

    mat = fitz.Matrix(zoom, zoom)
    if best_rect is not None:
        pix = page.get_pixmap(matrix=mat, clip=best_rect, alpha=False)
        pix.save(out_img_path)
        ok = os.path.getsize(out_img_path) >= 10000
        doc.close(); return ok

    # Fallback: region below heading
    y0 = max(pr.y0, min(pr.y1, y_bottom))
    clip = fitz.Rect(pr.x0, y0, pr.x1, pr.y1)
    pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
    pix.save(out_img_path)
    ok = os.path.getsize(out_img_path) >= 10000
    doc.close(); return ok

# ===================== PIPELINE =====================

def pipeline(image_bgr, weights_path):
    # 1) ACCESS (required)
    access = find_access_point(image_bgr)
    if access is None:
        raise RuntimeError("ACCESS not found in image.")

    # 2) Detect cautions (YOLO + OCR fallback), merge
    y_boxes = detect_caution_yolo(image_bgr, weights_path, conf=0.28, scales=(1.0, 1.6, 2.2))
    o_boxes = detect_caution_ocr(image_bgr, min_conf=0.35, scales=(1.4, 1.8, 2.2))
    boxes   = merge_yolo_ocr(y_boxes, o_boxes, iou_thr=0.25)

    # 3) Drop legend row (bottom band) so thumbnails aren’t counted
    H, W = image_bgr.shape[:2]
    boxes = [b for b in boxes if ((b[1]+b[3])*0.5) < H * 0.82]

    # 4) Order ALL boxes from ACCESS via nearest-neighbor tour
    ordered = order_boxes_nearest(boxes, access['center'])
    print(access,"ooooossdsdsf",ordered)
    return {"access": access, "ordered_boxes": ordered, "count": len(ordered)}

# ===================== MAIN =====================

def mainly(DEFAULT_PDF):
    print(DEFAULT_PDF,"lllllkk___")
    
    # ---- set these paths (once) ----
    # DEFAULT_PDF     = r"C:\Users\lenovo\Desktop\caution2k-20250818T051503Z-1-001\uploads\10087226_36504_CLL05824_CLL25824_RFSSR_04-22-2025.pdf"
    # DEFAULT_WEIGHTS = r"C:\Users\Admin\OneDrive\Documents\Desktop\caution2k-20250818T051503Z-1-001\best.pt"

    # EXTRACT_DIR = r"C:\Users\Admin\OneDrive\Documents\Desktop\caution2k-20250818T051503Z-1-001\extracted"
    # EXTRACT_IMG = os.path.join(EXTRACT_DIR, "site_scale_map_2.png")

    # OUTPUT_DIR  = r"C:\Users\Admin\OneDrive\Documents\Desktop\caution2k-20250818T051503Z-1-001\results"
    # OUTPUT_IMG  = os.path.join(OUTPUT_DIR, "site_scale_map_2_debug.jpg")
    BASE_DIR = os.getcwd()
    # DEFAULT_WEIGHTS = r"C:\Users\lenovo\Desktop\caution2k-20250818T051503Z-1-001\best.pt"
    # EXTRACT_DIR = r"C:\Users\lenovo\Desktop\caution2k-20250818T051503Z-1-001\extracted"
    DEFAULT_WEIGHTS = os.path.join(BASE_DIR, "best.pt")
    EXTRACT_DIR = os.path.join(BASE_DIR, "extracted")
    EXTRACT_IMG = os.path.join(EXTRACT_DIR, "site_scale_map_2.png")
    
    import random
    file_name = f"site_scale_{random.randint(0,987654321)}"
    file_path=f"results/{file_name}.jpg"
    OUTPUT_DIR  = f"{os.getcwd()}/results"
    print(OUTPUT_DIR,"qwet222")
    #C:\Users\lenovo\Desktop\caution2k-20250818T051503Z-1-001\results
    OUTPUT_IMG  = os.path.join(OUTPUT_DIR, f"{file_name}.jpg")

    # 1) PDF → image under 2nd "Site Scale Map"
    ok = extract_second_site_scale_map(DEFAULT_PDF, EXTRACT_IMG, keyword="Site Scale Map", margin_px=20, zoom=2.0)
    if not ok:
        print("[ERROR] Could not extract region under the 2nd 'Site Scale Map'.", file=sys.stderr)
        sys.exit(1)

    # 2) Run pipeline on extracted image
    img = cv2.imread(EXTRACT_IMG)
    if img is None:
        print(f"[ERROR] Could not read extracted image: {EXTRACT_IMG}", file=sys.stderr)
        sys.exit(1)

    res = pipeline(img, weights_path=DEFAULT_WEIGHTS)
    print(res,"res1")
    # 3) Print only the caution count
    print(res["count"],"res2")

    # 4) Save visualization
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    vis = draw_debug_with_path(img, res["access"], res["ordered_boxes"], show_indices=True, path_thickness=2)
    cv2.imwrite(OUTPUT_IMG, vis)
    last_file = files_collection.find_one(sort=[("_id", -1)])
    if not last_file:
        return {"error": "No uploaded file found"}
    files_collection.update_one(
            {"_id": last_file["_id"]},
            {"$set": {
                "mapping":file_path
            }}
        )
if __name__ == "__main__":
    main()

# access_pdf_to_detection.py
import cv2
import numpy as np
import re
import sys
import os

# ---------- PDF ----------
import fitz  # PyMuPDF

# ---------- YOLO ----------
from ultralytics import YOLO

# ---------- OCR ----------
import easyocr
OCR = easyocr.Reader(['en'], gpu=False)  # init once


# ===================== OCR UTILITIES =====================

def ocr_boxes(img, min_conf=0.45):
    """Return list of (text, conf, (x1,y1,x2,y2)) using EasyOCR on BGR/RGB image."""
    results = OCR.readtext(img)
    out = []
    for box, text, conf in results:
        if float(conf) < min_conf:
            continue
        xs = [p[0] for p in box]; ys = [p[1] for p in box]
        x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        out.append((text.strip(), float(conf), (x1, y1, x2, y2)))
    return out


def find_access_point(img):
    """
    Find the 'Access' label robustly and return {'center':(cx,cy), 'box':(x1,y1,x2,y2)}.
    Uses multi-scale OCR + contrast boost + fuzzy matching.
    """
    from difflib import SequenceMatcher

    def norm(txt):
        return re.sub(r'[^A-Z0-9]', '', (txt or '').upper())

    def sim_to_access(txt):
        s = norm(txt)
        if not s:
            return 0.0
        return SequenceMatcher(None, s, "ACCESS").ratio()

    def enhance(bgr):
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        L = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(L)
        return cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2BGR)

    scales = [1.6, 2.0, 2.4]
    cands = []
    for S in scales:
        up = cv2.resize(img, None, fx=S, fy=S, interpolation=cv2.INTER_LINEAR)
        up = enhance(up)
        for box, text, conf in OCR.readtext(up):
            score = sim_to_access(text)
            if score < 0.72 and norm(text) != "ACCESS":
                continue
            xs = [p[0] for p in box]; ys = [p[1] for p in box]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            x1o, y1o, x2o, y2o = int(x1 / S), int(y1 / S), int(x2 / S), int(y2 / S)
            cx, cy = (x1o + x2o) // 2, (y1o + y2o) // 2
            cands.append({
                "center": (cx, cy),
                "box": (x1o, y1o, x2o, y2o),
                "conf": float(conf),
                "score": float(score),
                "exact": norm(text) == "ACCESS",
                "x": cx
            })
        if cands:
            break
    if not cands:
        return None
    cands.sort(key=lambda c: (0 if c["exact"] else 1, c["x"], -c["score"], -c["conf"]))
    best = cands[0]
    return {"center": best["center"], "box": best["box"]}


def find_antennas(img, max_n=30):
    ants = []
    for text, conf, box in ocr_boxes(img, min_conf=0.35):
        t = text.strip().upper()
        t = re.sub(r'\s+', '', t).replace('I', '1').replace('L', '1')
        m = re.match(r'^A([0-9]{1,3})', t)
        if m:
            x1, y1, x2, y2 = box
            ants.append({
                'name': f"A{int(m.group(1))}",
                'center': ((x1+x2)//2, (y1+y2)//2),
                'box': (x1,y1,x2,y2),
                'conf': conf
            })
    best = {}
    for a in ants:
        k = a['name']
        if (k not in best) or (a['conf'] > best[k]['conf']):
            best[k] = a
    ants = [best[k] for k in sorted(best.keys(), key=lambda s: int(re.findall(r'\d+', s)[0]))]
    return ants[:max_n]


# ===================== DIRECTION / CORRIDOR =====================

def compute_travel_vector(access_pt, antennas):
    if not access_pt:
        return None
    if not antennas:
        return (1.0, 0.0)
    ax, ay = access_pt['center']
    xs = [a['center'][0] for a in antennas]
    ys = [a['center'][1] for a in antennas]
    cx, cy = float(np.mean(xs)), float(np.mean(ys))
    v = np.array([cx-ax, cy-ay], dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-6: return (1.0, 0.0)
    return (v / n)


def make_corridor_mask(h, w, access_pt, vhat, width_px=260):
    if access_pt is None or vhat is None:
        return np.ones((h,w), dtype=bool)
    ax, ay = access_pt['center']
    vx, vy = vhat
    nx, ny = -vy, vx
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    dist = np.abs((X-ax)*nx + (Y-ay)*ny)
    ahead = (X-ax)*vx + (Y-ay)*vy >= 0
    return (dist <= width_px/2) & ahead


# ===================== DETECT (YOLO + OCR fallback) =====================

def _iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    iw = max(0, x2 - x1 + 1); ih = max(0, y2 - y1 + 1)
    inter = iw * ih
    a1 = (b1[2]-b1[0]+1) * (b1[3]-b1[1]+1)
    a2 = (b2[2]-b2[0]+1) * (b2[3]-b2[1]+1)
    denom = a1 + a2 - inter
    return (inter / denom) if denom > 0 else 0.0

def _nms_merge(boxes, iou_thr=0.50):
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    kept = []
    for b in boxes:
        if all(_iou(b, k) < iou_thr for k in kept):
            kept.append(b)
    return kept

_YOLO_MODEL = None
_YOLO_PATH  = None

def detect_caution_yolo(img_bgr, weights_path, conf=0.20, imgsz=None, scales=(1.0, 1.6, 2.2)):
    """
    Multi-scale YOLO to catch small caution signs.
    Returns boxes in ORIGINAL image coords: [x1,y1,x2,y2,score]
    """
    global _YOLO_MODEL, _YOLO_PATH
    if (_YOLO_MODEL is None) or (_YOLO_PATH != weights_path):
        _YOLO_MODEL = YOLO(weights_path)
        _YOLO_PATH  = weights_path

    H, W = img_bgr.shape[:2]
    if imgsz is None:
        long_side = max(H, W)
        imgsz = int(np.clip(((long_side + 31)//32)*32, 960, 1536))

    all_boxes = []
    for S in scales:
        src = img_bgr if abs(S-1.0) < 1e-3 else cv2.resize(img_bgr, None, fx=S, fy=S, interpolation=cv2.INTER_LINEAR)
        res = _YOLO_MODEL.predict(source=src, imgsz=imgsz, conf=conf, agnostic_nms=True, verbose=False)[0]
        if res.boxes is None:
            continue
        for b in res.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().numpy().tolist())
            score = float(b.conf[0].cpu().numpy())
            if abs(S-1.0) >= 1e-3:
                x1 = int(x1 / S); y1 = int(y1 / S); x2 = int(x2 / S); y2 = int(y2 / S)
            x1 = max(0, min(W-1, x1)); x2 = max(0, min(W-1, x2))
            y1 = max(0, min(H-1, y1)); y2 = max(0, min(H-1, y2))
            all_boxes.append([x1, y1, x2, y2, score])
    return _nms_merge(all_boxes, iou_thr=0.5)

def _enhance_for_ocr(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    L = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(L)
    return cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2BGR)

def detect_caution_ocr(img_bgr, min_conf=0.30, scales=(1.4, 1.8, 2.2)):
    """Multi-scale OCR to catch tiny 'CAUTION' headers. Returns [x1,y1,x2,y2,score]."""
    H, W = img_bgr.shape[:2]
    out = []
    for S in scales:
        up = cv2.resize(img_bgr, None, fx=S, fy=S, interpolation=cv2.INTER_LINEAR)
        up = _enhance_for_ocr(up)
        for box, text, conf in OCR.readtext(up):
            if float(conf) < min_conf:
                continue
            if not re.search(r'\bCAUTION\b', (text or ''), flags=re.I):
                continue
            xs = [p[0] for p in box]; ys = [p[1] for p in box]
            x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
            if abs(S-1.0) >= 1e-3:
                x1 = int(x1 / S); y1 = int(y1 / S); x2 = int(x2 / S); y2 = int(y2 / S)
            x1 = max(0, min(W-1, x1)); x2 = max(0, min(W-1, x2))
            y1 = max(0, min(H-1, y1)); y2 = max(0, min(H-1, y2))
            out.append([x1, y1, x2, y2, float(conf)])
    return _nms_merge(out, iou_thr=0.4)

def merge_detections(yolo_boxes, ocr_boxes, iou_thr=0.25):
    """Union YOLO + OCR, merging overlaps by higher score."""
    out = list(yolo_boxes)
    for b in ocr_boxes:
        keep = True
        for i, a in enumerate(out):
            if _iou(a, b) >= iou_thr:
                if b[4] > a[4]:
                    out[i] = b
                keep = False
                break
        if keep:
            out.append(b)
    return out


# ===================== FILTER + ORDER =====================

def filter_boxes_to_corridor(boxes, corridor_mask, min_frac=0.35):
    kept = []
    H, W = corridor_mask.shape
    for b in boxes:
        x1,y1,x2,y2,score = b
        x1,y1,x2,y2 = map(int, (x1,y1,x2,y2))
        x1 = max(0, min(W-1, x1)); x2 = max(0, min(W-1, x2))
        y1 = max(0, min(H-1, y1)); y2 = max(0, min(H-1, y2))
        sub = corridor_mask[y1:y2+1, x1:x2+1]
        if sub.size == 0:
            continue
        frac = float(np.count_nonzero(sub)) / float(sub.size)
        if frac >= min_frac:
            kept.append([x1,y1,x2,y2,score])
    return kept

def _order_ahead_in_cone(boxes, origin_xy, vhat, max_angle_deg=60.0):
    """
    Keep boxes that are AHEAD of origin along vhat and within a wide cone.
    Then order them by projection distance.
    boxes: [x1,y1,x2,y2,score]
    """
    if not boxes:
        return []

    import math
    ox, oy = origin_xy
    vx, vy = vhat if vhat is not None else (1.0, 0.0)
    vnorm = math.hypot(vx, vy)
    if vnorm < 1e-6:
        vx, vy, vnorm = 1.0, 0.0, 1.0
    vx /= vnorm; vy /= vnorm

    max_cos = math.cos(math.radians(max_angle_deg))

    kept = []
    for b in boxes:
        x1, y1, x2, y2, s = b
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        dx = cx - ox
        dy = cy - oy
        proj = dx * vx + dy * vy         # distance ahead along vhat
        if proj <= 0:
            continue                     # behind ACCESS
        d = math.hypot(dx, dy)
        if d < 1e-6:
            continue
        cosang = (dx * vx + dy * vy) / d # = proj/d
        if cosang >= max_cos:            # inside cone
            kept.append((proj, b))

    kept.sort(key=lambda t: t[0])        # sort by distance ahead
    return [b for _, b in kept]


def order_boxes_along_vector(boxes, vhat, origin_xy):
    if not boxes:
        return []
    ox, oy = origin_xy
    vx, vy = vhat if vhat is not None else (1.0, 0.0)
    projs = []
    for i, (x1,y1,x2,y2,_) in enumerate(boxes):
        cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
        projs.append(((cx-ox)*vx + (cy-oy)*vy, i))
    projs.sort()
    return [boxes[i] for _, i in projs]


# ===================== PIPELINE =====================

def pipeline(image_bgr, yolo_weights):
    H, W = image_bgr.shape[:2]

    # 1) Find ACCESS (must exist)
    access_pt = find_access_point(image_bgr)
    if access_pt is None:
        raise RuntimeError("ACCESS text not found in the image — cannot start traversal.")

    # 2) Optional antennas (only to estimate direction)
    antennas = find_antennas(image_bgr)

    # 3) Detect ALL cautions first (YOLO multi-scale + OCR fallback), then merge
    boxes_y = detect_caution_yolo(image_bgr, yolo_weights, conf=0.20, imgsz=None, scales=(1.0, 1.6, 2.2))
    boxes_o = detect_caution_ocr(image_bgr, min_conf=0.30, scales=(1.4, 1.8, 2.2))
    boxes   = merge_detections(boxes_y, boxes_o, iou_thr=0.25)

    # 4) Choose travel direction:
    #    prefer ACCESS -> antennas centroid; if no antennas, use ACCESS -> cautions centroid (rightward bias)
    if antennas:
        vhat = compute_travel_vector(access_pt, antennas)
    else:
        if boxes:
            xs = [(b[0]+b[2]) * 0.5 for b in boxes]
            ys = [(b[1]+b[3]) * 0.5 for b in boxes]
            cx, cy = float(np.mean(xs)), float(np.mean(ys))
            ax, ay = access_pt['center']
            v = np.array([cx - ax, cy - ay], dtype=float)
            n = np.linalg.norm(v)
            vhat = (v / n) if n > 1e-6 else (1.0, 0.0)
        else:
            vhat = (1.0, 0.0)

    # 5) NO corridor mask — just a wide cone ahead of ACCESS (less strict)
    #    if you still want stricter/looser gating, tweak max_angle_deg.
    ordered = _order_ahead_in_cone(
        boxes,
        origin_xy=access_pt['center'],
        vhat=vhat,
        max_angle_deg=60.0   # try 70–80 if needed, or 45 for stricter
    )

    return {
        'access': access_pt,
        'antennas': antennas,
        'vhat': vhat,
        'corridor_mask': None,        # not used anymore
        'ordered_boxes': ordered,
        'count': len(ordered)
    }

# ===================== DRAWING =====================

def _box_center(b):
    x1,y1,x2,y2,_ = b
    return int((x1+x2)//2), int((y1+y2)//2)

def draw_debug_with_path(image_bgr, res, show_indices=True, path_thickness=2):
    img = image_bgr.copy()
    H, W = img.shape[:2]

    # Access (must exist)
    x1,y1,x2,y2 = res['access']['box']
    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,255), 2)
    ax,ay = res['access']['center']
    cv2.circle(img, (ax,ay), 6, (0,255,255), -1)
    cv2.putText(img, 'ACCESS', (x1, max(0,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    # Cautions and path
    route_points = [(ax, ay)]
    for idx, b in enumerate(res['ordered_boxes'], start=1):
        x1,y1,x2,y2,score = b
        cx, cy = _box_center(b)
        route_points.append((cx, cy))
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,165,255), 2)
        if show_indices:
            cv2.putText(img, f"{idx}", (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,165,255), 2)

    if len(route_points) >= 2:
        cv2.polylines(img, [np.array(route_points, dtype=np.int32)], False, (0,255,0), path_thickness)
        for i in range(len(route_points)-1):
            cv2.arrowedLine(img, route_points[i], route_points[i+1], (0,255,0), path_thickness, tipLength=0.1)
    return img


# ===================== PDF → IMAGE (2nd "Site Scale Map") =====================

def extract_second_site_scale_map(pdf_path, out_img_path, keyword="Site Scale Map", margin_px=20, zoom=2.0):
    """
    Find the 2nd occurrence of `keyword`; prefer the largest image block below it.
    Fallback: render region below the heading.
    """
    if not os.path.exists(pdf_path):
        return False
    os.makedirs(os.path.dirname(out_img_path), exist_ok=True)
    doc = fitz.open(pdf_path)

    seen = set(); hits = []
    # A) text search
    for pno in range(len(doc)):
        page = doc[pno]
        for q in (keyword, keyword.upper(), keyword.lower()):
            try:
                rects = page.search_for(q) or []
            except Exception:
                rects = []
            for r in rects:
                key = (pno, int(round(r.y1 * 1000)))
                if key not in seen:
                    seen.add(key); hits.append((pno, r.y1))

    # B) OCR top 35% if needed
    if len(hits) < 2:
        for pno in range(len(doc)):
            page = doc[pno]; pr = page.rect
            clip = fitz.Rect(pr.x0, pr.y0, pr.x1, pr.y0 + 0.35 * pr.height)
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
            import numpy as _np
            rgb = _np.frombuffer(pix.samples, dtype=_np.uint8).reshape(pix.height, pix.width, pix.n)
            if rgb.shape[2] > 3: rgb = rgb[:, :, :3]
            for box, txt, conf in OCR.readtext(rgb):
                if float(conf) < 0.45: continue
                norm = re.sub(r"\s+", " ", (txt or '').lower()).strip()
                if "site scale map" in norm:
                    y_px = max(pt[1] for pt in box) + margin_px
                    y_page = (y_px / zoom) + clip.y0
                    key = (pno, int(round(y_page * 1000)))
                    if key not in seen:
                        seen.add(key); hits.append((pno, y_page))
                        if len(hits) >= 2: break
            if len(hits) >= 2: break

    if len(hits) < 2:
        doc.close(); return False

    hits.sort(key=lambda t: (t[0], t[1]))
    pno, y_bottom = hits[1]  # strictly 2nd
    page = doc[pno]; pr = page.rect
    raw = page.get_text("rawdict")

    biggest_bbox = None; biggest_area = 0.0
    page_area = max(1e-6, pr.width * pr.height)
    for block in raw.get("blocks", []):
        if block.get("type") != 1:  # images
            continue
        bbox = block.get("bbox")
        if not bbox: continue
        x0, y0, x1, y1 = bbox
        if y0 <= y_bottom:  # must be below heading
            continue
        area = max(0.0, (x1 - x0) * (y1 - y0))
        if (area / page_area) < 0.05:
            continue
        if area > biggest_area:
            biggest_area = area
            biggest_bbox = fitz.Rect(x0, y0, x1, y1)

    mat = fitz.Matrix(zoom, zoom)
    if biggest_bbox is not None:
        pix = page.get_pixmap(matrix=mat, clip=biggest_bbox, alpha=False)
        pix.save(out_img_path)
        ok = os.path.getsize(out_img_path) >= 10000
        doc.close(); return ok

    # fallback: region below heading
    y0 = max(pr.y0, min(pr.y1, y_bottom))
    clip = fitz.Rect(pr.x0, y0, pr.x1, pr.y1)
    pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
    pix.save(out_img_path)
    ok = os.path.getsize(out_img_path) >= 10000
    doc.close(); return ok


# ===================== MAIN =====================

def main(DEFAULT_PDF):
    # ---- set paths here ----
    # DEFAULT_PDF     = r"C:\Users\lenovo\Desktop\caution2k-20250818T051503Z-1-001\uploads\10087226_36504_CLL05824_CLL25824_RFSSR_04-22-2025.pdf"
    DEFAULT_WEIGHTS = r"C:\Users\lenovo\Desktop\caution2k-20250818T051503Z-1-001\best.pt"
    EXTRACT_DIR = r"C:\Users\lenovo\Desktop\caution2k-20250818T051503Z-1-001\extracted"
    EXTRACT_IMG = os.path.join(EXTRACT_DIR, "site_scale_map_2.png")
    OUTPUT_DIR  = r"C:\Users\lenovo\Desktop\caution2k-20250818T051503Z-1-001\results"
    OUTPUT_IMG  = os.path.join(OUTPUT_DIR, "site_scale_map_2_debug.jpg")

    ok = extract_second_site_scale_map(DEFAULT_PDF, EXTRACT_IMG, keyword="Site Scale Map", margin_px=20, zoom=2.0)
    if not ok:
        print("[ERROR] Could not extract the region under the 2nd 'Site Scale Map' heading.", file=sys.stderr)
        sys.exit(1)

    img = cv2.imread(EXTRACT_IMG)
    if img is None:
        print(f"[ERROR] Could not read extracted image: {EXTRACT_IMG}", file=sys.stderr)
        sys.exit(1)

    try:
        res = pipeline(img, yolo_weights=DEFAULT_WEIGHTS)
    except RuntimeError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    print("\n=== RESULTS ===")
    if res['access']:
        print("Access at:", res['access']['center'])
    print("Antennas found:", [a['name'] for a in res['antennas']])
    print("Caution signs count (travel order):", res['count'])

    dbg = draw_debug_with_path(img, res, show_indices=True, path_thickness=2)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if cv2.imwrite(OUTPUT_IMG, dbg):
        print("Output saved to:", OUTPUT_IMG)
    else:
        print("[ERROR] cv2.imwrite failed. Check the output path/extension.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

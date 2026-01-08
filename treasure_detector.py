import cv2
import os
import time

MATCH_THRESHOLD = 0.40

ROI_X_START_FRAC = 0.50
ROI_X_END_FRAC   = 0.82
ROI_Y_START_FRAC = 0.84
ROI_Y_END_FRAC   = 0.97

TEMPLATES_DIR = "treasures"

def load_templates():
    templates = []
    for name in os.listdir(TEMPLATES_DIR):
        if name.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(TEMPLATES_DIR, name)
            img = cv2.imread(path)
            if img is not None:
                templates.append((name, img))
    return templates

TEMPLATES = load_templates()

def crop_roi(img):
    H, W = img.shape[:2]
    xs = int(W * ROI_X_START_FRAC)
    xe = int(W * ROI_X_END_FRAC)
    ys = int(H * ROI_Y_START_FRAC)
    ye = int(H * ROI_Y_END_FRAC)
    return img[ys:ye, xs:xe]

def detect_treasure(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        return None, 0.0

    roi = crop_roi(img)

    best_score = 0.0
    best_name = None

    for name, tmpl in TEMPLATES:
        if roi.shape[0] < tmpl.shape[0] or roi.shape[1] < tmpl.shape[1]:
            continue
        res = cv2.matchTemplate(roi, tmpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, _ = cv2.minMaxLoc(res)
        if score > best_score:
            best_score = score
            best_name = name

    if best_score >= MATCH_THRESHOLD:
        return best_name, best_score

    return None, best_score


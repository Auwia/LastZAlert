import cv2
from lastz_treasure_watcher import (
    TREASURE_ROI,
    MATCH_THRESHOLD_TREASURE,
    load_templates_from_dir,
    crop_roi,
    match_any,
)

IMG_PATH = "test.png"   # <-- metti qui il tuo screenshot

#templates = load_templates_from_dir("treasures")
templates = load_templates_from_dir("hq_upgrade")

img = cv2.imread(IMG_PATH)
if img is None:
    raise RuntimeError("Immagine non caricata")

roi, coords = crop_roi(img, TREASURE_ROI)
name, score, loc, hw = match_any(roi, templates)

print("ROI coords:", coords)
print("Match:", name)
print("Score:", score)
print("Threshold:", MATCH_THRESHOLD_TREASURE)

cv2.imwrite("debug_test_roi.png", roi)


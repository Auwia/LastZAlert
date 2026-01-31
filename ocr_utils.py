import cv2
import pytesseract
import re
from bot_utils import debug_save

DEBUG = True

def read_timer_seconds(img):
    import cv2, re, pytesseract

    # 1. upscale forte (fondamentale)
    img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    # 2. converti in HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 3. estrai SOLO testo chiaro (bianco/giallo)
    # range largo apposta
    mask = cv2.inRange(
        hsv,
        (0, 0, 200),    # bassa saturazione, alto valore
        (180, 60, 255)
    )

    # 4. pulizia
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 5. inverti per OCR
    mask = cv2.bitwise_not(mask)

    if DEBUG:
        debug_save(mask, "timer_mask")

    # 6. OCR
    text = pytesseract.image_to_string(
        mask,
        config="--psm 7 -c tessedit_char_whitelist=0123456789:"
    ).strip()

    if DEBUG:
        print(f"[OCR] raw text = '{text}'")

    parts = re.findall(r"\d+", text)

    if len(parts) == 3:
        h, m, s = map(int, parts)
        return h * 3600 + m * 60 + s

    if len(parts) == 2:
        m, s = map(int, parts)
        return m * 60 + s

    return None


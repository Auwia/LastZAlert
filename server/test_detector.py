from treasure_detector import detect_treasure

IMG = "test_heli.png"

name, score = detect_treasure(IMG)

print("RISULTATO:")
print("  name =", name)
print("  score =", score)


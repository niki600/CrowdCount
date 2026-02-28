import cv2
import json
import os
from datetime import datetime

ZONE_FILE = "zones.json"

drawing = False
start_point = None
temp_rect = None
zones = []
fullscreen = False

# ---------- Load saved zones ----------
if os.path.exists(ZONE_FILE):
    with open(ZONE_FILE, "r") as f:
        zones = json.load(f)

# ---------- Save zones ----------
def save_zones():
    with open(ZONE_FILE, "w") as f:
        json.dump(zones, f, indent=4)

# ---------- Mouse callback ----------
def draw_zone(event, x, y, flags, param):
    global drawing, start_point, temp_rect, zones

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        temp_rect = (start_point[0], start_point[1], x, y)


    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = start_point
        x2, y2 = x, y

        if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
            zone_data = {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            zones.append(zone_data)
            save_zones()
            print(f"Zone {len(zones)} saved at {zone_data['created_at']}")

        temp_rect = None

# ---------- Open camera ----------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera not accessible")
    exit()

cv2.namedWindow("Live Feed")
cv2.setMouseCallback("Live Feed", draw_zone)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame")
        break

    # ---------- Draw saved zones ----------
    for i, z in enumerate(zones):
        cv2.rectangle(frame, (z["x1"], z["y1"]), (z["x2"], z["y2"]), (0, 255, 0), 2)

        # Zone label
        cv2.putText(frame, f"Zone {i+1}", (z["x1"], z["y1"] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # Count placeholder
        cv2.putText(frame, "Count: 0",
                    (z["x1"] + 5, z["y1"] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # ---------- Draw temp rectangle ----------
    if temp_rect:
        x1, y1, x2, y2 = temp_rect
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

    # ---------- Instruction Overlay ----------
    instructions = [
        "Draw: Mouse drag",
        "d: Delete last zone",
        "r: Reset all zones",
        "p: Save screenshot",
        "f: Fullscreen",
        "q: Quit"
    ]

    for i, text in enumerate(instructions):
        cv2.putText(frame, text, (10, 20 + i*20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

    cv2.imshow("Live Feed", frame)

    key = cv2.waitKey(1) & 0xFF

    # ---------- Controls ----------
    if key == ord('q'):
        break

    elif key == ord('d'):
        if zones:
            zones.pop()
            save_zones()
            print("Last zone deleted")

    elif key == ord('r'):
        zones = []
        save_zones()
        print("All zones cleared")

    elif key == ord('p'):
        filename = f"screenshot_{datetime.now().strftime('%H%M%S')}.png"
        cv2.imwrite(filename, frame)
        print("Screenshot saved:", filename)

    elif key == ord('f'):
        fullscreen = not fullscreen
        if fullscreen:
            cv2.setWindowProperty("Live Feed",
                                  cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty("Live Feed",
                                  cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_NORMAL)

cap.release()
cv2.destroyAllWindows()
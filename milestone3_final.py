import cv2
import json
import math
import csv
from ultralytics import YOLO

# -------- Load YOLO --------
model = YOLO("yolov8n.pt")

# -------- Load Zones --------
ZONE_FILE = "zones.json"
zones = []

try:
    with open(ZONE_FILE, "r") as f:
        zones = json.load(f)
    print("Zones loaded:", len(zones))
except:
    print("No zones found")

# -------- CSV --------
csv_file = "count_data.csv"

# -------- Camera --------
cap = cv2.VideoCapture(0)   # IMPORTANT: use 0

if not cap.isOpened():
    print("Camera not accessible")
    exit()

# -------- Tracking --------
person_id = 0
tracks = {}
counted_ids = set()

entry_count = 0
exit_count = 0

zone_counts = {i: 0 for i in range(len(zones))}

line_y = 250

# -------- Helper --------
def get_center(x1, y1, x2, y2):
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def inside_zone(cx, cy, zone):
    return zone["x1"] < cx < zone["x2"] and zone["y1"] < cy < zone["y2"]

print("Milestone-3 running... Press q to exit")

# -------- Main Loop --------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for speed
    frame = cv2.resize(frame, (640, 480))

    # -------- YOLO Detection --------
    results = model(frame, verbose=False)

    detections = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls == 0:   # person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center = get_center(x1, y1, x2, y2)
                detections.append((x1, y1, x2, y2, center))

    # -------- Tracking --------
    new_tracks = {}

    for det in detections:
        x1, y1, x2, y2, center = det

        matched_id = None
        for pid, prev_center in tracks.items():
            if distance(center, prev_center) < 50:
                matched_id = pid
                break

        if matched_id is None:
            person_id += 1
            matched_id = person_id

        new_tracks[matched_id] = center

        # Entry / Exit (count once)
        if matched_id not in counted_ids:
            if center[1] < line_y:
                entry_count += 1
                counted_ids.add(matched_id)
            elif center[1] > line_y:
                exit_count += 1
                counted_ids.add(matched_id)

        # Zone check (count once per ID)
        for i, zone in enumerate(zones):
            if inside_zone(center[0], center[1], zone):
                zone_counts[i] += 1

        # Draw
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"ID {matched_id}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    tracks = new_tracks

    # -------- Draw Zones --------
    for i, zone in enumerate(zones):
        cv2.rectangle(frame,
                      (zone["x1"], zone["y1"]),
                      (zone["x2"], zone["y2"]),
                      (255,0,0), 2)

        cv2.putText(frame,
                    f"Zone {i+1}: {zone_counts[i]}",
                    (zone["x1"], zone["y1"] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255,0,0), 2)

    # -------- Virtual Line --------
    cv2.line(frame, (0, line_y), (640, line_y), (0,0,255), 2)

    # -------- Dashboard --------
    cv2.putText(frame, f"Entry: {entry_count}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(frame, f"Exit: {exit_count}", (20,70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(frame, f"Active: {len(tracks)}", (20,100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    # -------- Show --------
    cv2.imshow("Milestone-3 Final", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        # Save CSV
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([entry_count, exit_count])
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import json
import math
import csv
from ultralytics import YOLO
import time
import matplotlib.pyplot as plt

# ---------------- Camera ----------------
cap = cv2.VideoCapture(0)
time.sleep(2)

if not cap.isOpened():
    print("Camera not accessible")
    exit()

print("Camera started successfully")

# ---------------- Load YOLO ----------------
model = YOLO("yolov8n.pt")

# ---------------- Load Zones ----------------
zones = []
try:
    with open("zones.json", "r") as f:
        zones = json.load(f)
    print("Zones Loaded:", len(zones))
except:
    print("No zones found")

# ---------------- CSV ----------------
csv_file = "count_data.csv"

# ---------------- Variables ----------------
person_id = 0
tracks = {}
zone_tracked = {}

entry_count = 0
exit_count = 0

zone_counts = {f"Zone {i+1}": 0 for i in range(len(zones))}

line_y = 250

# ---------------- Helper ----------------
def get_center(x1,y1,x2,y2):
    return int((x1+x2)/2), int((y1+y2)/2)

def distance(p1,p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def inside_zone(cx,cy,zone):
    return zone["x1"] < cx < zone["x2"] and zone["y1"] < cy < zone["y2"]

print("Milestone-4 running... Press Q to exit")

# ---------------- Main Loop ----------------
while True:

    ret, frame = cap.read()

    if not ret:
        continue

    frame = cv2.resize(frame,(640,480))

    results = model(frame, verbose=False)

    detections = []

    for r in results:
        for box in r.boxes:

            cls = int(box.cls[0])

            if cls == 0:   # person class

                x1,y1,x2,y2 = map(int,box.xyxy[0])

                center = get_center(x1,y1,x2,y2)

                detections.append((x1,y1,x2,y2,center))

    # -------- Tracking --------
    new_tracks = {}

    for det in detections:

        x1,y1,x2,y2,center = det

        matched_id = None

        for pid,prev_center in tracks.items():

            if distance(center,prev_center) < 50:
                matched_id = pid
                break

        if matched_id is None:
            person_id += 1
            matched_id = person_id

        # -------- Entry Exit Logic --------
        if matched_id in tracks:

            prev_y = tracks[matched_id][1]
            curr_y = center[1]

            if prev_y < line_y and curr_y >= line_y:
                entry_count += 1

            elif prev_y > line_y and curr_y <= line_y:
                exit_count += 1

        new_tracks[matched_id] = center

        # -------- Zone Counting --------
        for i,zone in enumerate(zones):

            if inside_zone(center[0],center[1],zone):

                if matched_id not in zone_tracked:
                    zone_tracked[matched_id] = set()

                if i not in zone_tracked[matched_id]:

                    zone_counts[f"Zone {i+1}"] += 1
                    zone_tracked[matched_id].add(i)

        # -------- Draw Person --------
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

        cv2.putText(frame,f"ID {matched_id}",
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,(0,255,255),2)

    tracks = new_tracks

    # -------- Draw Zones --------
    for i,zone in enumerate(zones):

        cv2.rectangle(frame,
                      (zone["x1"],zone["y1"]),
                      (zone["x2"],zone["y2"]),
                      (255,0,0),2)

        cv2.putText(frame,
                    f"Zone {i+1}: {zone_counts[f'Zone {i+1}']}",
                    (zone["x1"],zone["y1"]-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,(255,0,0),2)

    # -------- Virtual Line --------
    cv2.line(frame,(0,line_y),(640,line_y),(0,0,255),2)

    # -------- Dashboard --------
    cv2.putText(frame,f"Entry: {entry_count}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,(0,255,255),2)

    cv2.putText(frame,f"Exit: {exit_count}",
                (20,70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,(0,255,255),2)

    cv2.putText(frame,f"Active: {len(tracks)}",
                (20,100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,(0,255,255),2)

    cv2.imshow("Milestone-4 CrowdCount",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):

        with open(csv_file,"a",newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Entry",entry_count,"Exit",exit_count])

        break

cap.release()
cv2.destroyAllWindows()

# -------- Crowd Trend Graph --------
plt.figure()
plt.title("Crowd Entry vs Exit Trend")
plt.bar(["Entry","Exit"],[entry_count,exit_count])
plt.xlabel("Crowd Movement")
plt.ylabel("Number of People")
plt.show()

# -------- Zone Usage Graph --------
zones_list = list(zone_counts.keys())
values = list(zone_counts.values())

plt.figure()
plt.title("Zone Usage Analysis")
plt.bar(zones_list,values)
plt.xlabel("Zones")
plt.ylabel("People Count")
plt.show()
import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)   # try 0 first

if not cap.isOpened():
    print("Camera not accessible")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame not received")
        break

    # Run detection
    results = model(frame, stream=True)

    person_count = 0

    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                cls = int(box.cls[0])

                # Class 0 = person
                if cls == 0:
                    person_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display count
    cv2.putText(frame, f"People: {person_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Milestone-2: People Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
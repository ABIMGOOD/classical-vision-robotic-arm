
import cv2
import numpy as np
import csv
import os

CAMERA_INDEX = 1
CSV_FILE = "red_objects.csv"

LABELS = {
    ord("1"): "container_red",
    ord("2"): "screwdriver",
    ord("3"): "safety_cutter",
}

FEATURE_NAMES = [
    "area",
    "width",
    "height",
    "aspect_ratio",
    "perimeter",
    "circularity",
    "solidity",
    "extent",
    "label",
]

# Create the CSV header if the file does not exist
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=FEATURE_NAMES
        )
        writer.writeheader()


def extract_features(contour):
    area = cv2.contourArea(contour)

    x, y, width, height = cv2.boundingRect(contour)

    perimeter = cv2.arcLength(contour, True)

    circularity = (
        4 * np.pi * area / (perimeter ** 2)
        if perimeter > 0 else 0
    )

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)

    solidity = (
        area / hull_area if hull_area > 0 else 0
    )

    box_area = width * height

    extent = (
        area / box_area if box_area > 0 else 0
    )

    return {
        "area": area,
        "width": width,
        "height": height,
        "aspect_ratio": width / height,
        "perimeter": perimeter,
        "circularity": circularity,
        "solidity": solidity,
        "extent": extent,
    }


cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError("Could not open camera")

print("Place ONE red object in view.")
print("1 = red container")
print("2 = screwdriver")
print("3 = safety cutter")
print("Q = quit")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not read camera frame")
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Red wraps around the HSV hue boundary
    mask1 = cv2.inRange(
        hsv, np.array([0, 70, 40]),
        np.array([10, 255, 255])
    )
    mask2 = cv2.inRange(
        hsv, np.array([170, 70, 40]),
        np.array([179, 255, 255])
    )

    red_mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((3, 3), np.uint8)
    red_mask = cv2.morphologyEx(
        red_mask, cv2.MORPH_OPEN, kernel
    )
    red_mask = cv2.morphologyEx(
        red_mask, cv2.MORPH_CLOSE, kernel
    )

    contours, _ = cv2.findContours(
        red_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Find the largest red contour
    candidates = [
        c for c in contours
        if cv2.contourArea(c) >= 100
    ]

    largest = (
        max(candidates, key=cv2.contourArea)
        if candidates else None
    )

    display = frame.copy()

    if largest is not None:
        x, y, w, h = cv2.boundingRect(largest)

        cv2.rectangle(
            display, (x, y), (x + w, y + h),
            (0, 255, 0), 2
        )

        cv2.putText(
            display, "Detected object",
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 255, 0), 2
        )

    cv2.imshow("Object data collection", display)
    cv2.imshow("Red mask", red_mask)

    key = cv2.waitKey(1) & 0xFF

    if key in LABELS and largest is not None:
        features = extract_features(largest)
        features["label"] = LABELS[key]

        with open(CSV_FILE, "a", newline="") as file:
            writer = csv.DictWriter(
                file, fieldnames=FEATURE_NAMES
            )
            writer.writerow(features)

        print("Saved:", features)

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

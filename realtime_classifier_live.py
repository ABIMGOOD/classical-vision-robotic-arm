
import cv2
import numpy as np
import joblib

# Load your trained model
saved = joblib.load("red_object_classifier.joblib")
model = saved["model"]
features = saved["features"]

# Open webcam
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    raise RuntimeError("Could not open camera index 1")


def extract_features(contour):
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    perimeter = cv2.arcLength(contour, True)

    circularity = (
        4 * np.pi * area / (perimeter ** 2)
        if perimeter > 0 else 0
    )

    hull_area = cv2.contourArea(cv2.convexHull(contour))
    solidity = area / hull_area if hull_area > 0 else 0

    extent = area / (w * h) if w * h > 0 else 0

    return {
        "area": area,
        "width": w,
        "height": h,
        "aspect_ratio": w / float(h),
        "perimeter": perimeter,
        "circularity": circularity,
        "solidity": solidity,
        "extent": extent,
    }


while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not read frame")
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Detect red: hue wraps around the HSV boundary
    lower_red1 = np.array([0, 70, 40])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 40])
    upper_red2 = np.array([179, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
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

    for contour in contours:
        if cv2.contourArea(contour) < 100:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        measurements = extract_features(contour)

        # Arrange measurements in the exact feature order
        sample = np.array([[
            measurements[name] for name in features
        ]])

        # Predict object identity
        prediction = model.predict(sample)[0]

        # Find the object's centre in image pixels
        moments = cv2.moments(contour)

        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
        else:
            cx = x + w // 2
            cy = y + h // 2

        # Draw box, centre and prediction
        cv2.rectangle(
            frame, (x, y), (x + w, y + h),
            (0, 255, 0), 2
        )

        cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

        cv2.putText(
            frame, prediction,
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 255, 0), 2
        )

        cv2.putText(
            frame, f"Centre: ({cx}, {cy})",
            (x, y + h + 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (255, 255, 0), 1
        )

    cv2.imshow("Real-Time Object Classifier", frame)
    cv2.imshow("Red Segmentation Mask", red_mask)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

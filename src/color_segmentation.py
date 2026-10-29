# COLOUR SEGMENTATION FOR RED AND YELLOW 


import cv2
import numpy as np

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break


    # Convert the camera image from BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Red shade range 1
    red_mask1 = cv2.inRange(
        hsv,
        np.array([0, 70, 40]),
        np.array([10, 255, 255])
    )

    # Red shade range 2
    red_mask2 = cv2.inRange(
        hsv,
        np.array([170, 70, 40]),
        np.array([179, 255, 255])
    )

    # Combine the two red hue ranges
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # Yellow container
    yellow_mask = cv2.inRange(
        hsv,
        np.array([20, 70, 40]),
        np.array([35, 255, 255])
    )

    # Display each mask separately
    cv2.imshow("Red mask", red_mask)
    cv2.imshow("Yellow mask", yellow_mask)
    cv2.imshow("Original image", frame)

    def show_hsv(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pixel = hsv[y, x]
            print(
                f"Pixel ({x}, {y}) ->"
                f"H = {pixel[0]}, "
                f"S = {pixel[1]}, "
                f"V = {pixel[2]}"
            )
    cv2.setMouseCallback("Original image", show_hsv)


    contours, _= cv2.findContours(
        red_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    for contour in contours:
        area = cv2.contourArea(contour)

        if area < 100:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        aspect_ratio = w / float(h)
        perimeter = cv2.arcLength(contour, True)


        circularity = (
            4 * np.pi * area / (perimeter ** 2)
        
        if perimeter > 0 else 0
        )

        hull_area = cv2.contourArea(
            cv2.convexHull(contour)
        )

        solidity = (
            area / hull_area if hull_area > 0 else 0
        )

        print(
        f"Area={area:.0f}, "
        f"Aspect ratio={aspect_ratio:.2f}, "
        f"Circularity={circularity:.2f}, "
        f"Solidity={solidity:.2f}"
    )



    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

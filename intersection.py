import cv2
import numpy as np
import matplotlib.pyplot as plt

def highlight_intersection(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) < 2:
        print("Error: Less than two contours found.")
        return

    # Assuming the two largest contours are the pencils
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

    # Create an empty mask
    mask = np.zeros_like(gray)

    # Draw the contours on the mask
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    # Find the intersection area
    intersection_mask = np.zeros_like(gray)
    cv2.drawContours(intersection_mask, contours, -1, 255, thickness=cv2.FILLED)
    intersection = cv2.bitwise_and(intersection_mask, intersection_mask, mask=mask)

    # Find contours of the intersection area
    intersection_contours, _ = cv2.findContours(intersection, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If no intersection contours found, exit
    if not intersection_contours:
        print("Error: No intersection area found.")
        return

    # Find the smallest contour which is likely to be the actual intersection point
    smallest_contour = min(intersection_contours, key=cv2.contourArea)

    # Highlight the intersection area
    x, y, w, h = cv2.boundingRect(smallest_contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Optional: Draw a circle at the center of the intersection area
    center_x, center_y = x + w // 2, y + h // 2
    cv2.circle(image, (center_x, center_y), 10, (0, 0, 255), -1)

    # Convert the image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    denoised_image = cv2.medianBlur(image_rgb, 3)

    return denoised_image
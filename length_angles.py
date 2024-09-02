import cv2
import matplotlib.pyplot as plt
import numpy as np

def calculate_lengths_and_angles(image_path):

  # Load the image
  image = cv2.imread(image_path)

  denoised_image = cv2.medianBlur(image, 3)

  # Edge detection
  edges = cv2.Canny(denoised_image, 50, 150, apertureSize=3)

  # Hough Line Transform to detect lines
  lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

  # Draw lines and calculate lengths
  lengths = []
  angles = []
  for line in lines:
      x1, y1, x2, y2 = line[0]
      cv2.line(denoised_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
      length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
      lengths.append(length)
      angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
      angles.append(angle)

  # Assuming lengths[0] is pencil A and lengths[1] is pencil B
  length_a_mm = lengths[0] * 0.1
  length_b_mm = lengths[1] * 0.1

  # Calculate angle between the two pencils
  angle_between_pencils = np.abs(angles[0] - angles[1])

  return length_a_mm, length_b_mm, angle_between_pencils

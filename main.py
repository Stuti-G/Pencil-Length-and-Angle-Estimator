
import cv2
import numpy as np
import matplotlib.pyplot as plt
from length_angles import calculate_lengths_and_angles
from intersection import highlight_intersection


img_path = '2_pencils.jpg'
image = cv2.imread(img_path)
intersected_image = highlight_intersection(img_path)
DI = cv2.medianBlur(image, 3)

length_a_mm,length_b_mm,angle_between_pencils = calculate_lengths_and_angles(img_path)

# Display the results
print(f"Length of Pencil A:{length_a_mm: .2f} mm")
print(f"Length of Pencil B:{length_b_mm: .2f} mm")
print(f"Angle between the pencils: {angle_between_pencils: .2f} degrees")

# Display the original, line-detected images and denoised.
plt.figure(figsize=(20,20))
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.subplot(1, 3, 2)
plt.title('Intersected Region')
plt.imshow(cv2.cvtColor(intersected_image, cv2.COLOR_BGR2RGB))

plt.subplot(1, 3 ,3)
plt.title('Denoised Image')
plt.imshow(cv2.cvtColor(DI, cv2.COLOR_BGR2RGB))


plt.savefig('output.png')
plt.show()


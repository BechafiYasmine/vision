
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the image
img = cv2.imread("usthb.jpg", cv2.IMREAD_COLOR)

if img is None:
    raise FileNotFoundError("Image couldn't be loaded. Check the file path.")

# 2. Get image dimensions
height, width = img.shape[:2]
print(f"Width: {width} pixels")
print(f"Height: {height} pixels")

# 3. Convert BGR to RGB for Matplotlib
imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 4. Display the original image
plt.figure(figsize=(8, 6))
plt.imshow(imgRGB)
plt.title("Original Image")
plt.xlabel(f"Width: {width} pixels")
plt.ylabel(f"Height: {height} pixels")
plt.show()

# 5. Convert BGR to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 6. Define the green color range
lower_green = np.array([35, 58, 50])
upper_green = np.array([85, 255, 255])

# 7. Create a mask for green pixels
mask = cv2.inRange(hsv, lower_green, upper_green)

# 8. Extract green areas from the original image
result = np.zeros_like(img)
result[mask > 0] = img[mask > 0]

# 9. Display the mask and extracted green areas
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(mask, cmap="gray")
plt.title("Green Color Mask")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title("Extracted Green Areas")
plt.axis("off")

plt.tight_layout()
plt.show()


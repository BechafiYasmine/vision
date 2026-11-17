import cv2
import math
import numpy as np

# 1. Load the image
img = cv2.imread("usthb.jpg", cv2.IMREAD_COLOR)

if img is None:
    raise FileNotFoundError("Image couldn't be loaded. Check the file path.")

# 2. Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


h, w, c = img.shape

# Niveaux de gris (moyenne des 3 canaux)
img_gris = np.mean(img, axis=2).astype("uint8")

# Noyau gaussien 7x7
sigma = 1.4

def gauss(x, y):
    return (1 / (2 * math.pi * sigma**2)) * math.exp(-(x*x + y*y) / (2 * sigma**2))

kernel_gauss = np.zeros((7, 7))
som = 0
for i in range(-3, 4):
    for j in range(-3, 4):
        val = gauss(i, j)
        kernel_gauss[i+3, j+3] = val
        som += val

kernel_gauss = kernel_gauss / som  # normalisation (somme = 1)

img_gauss = cv2.filter2D(img_gris, cv2.CV_8U, kernel_gauss)

# Noyau Laplacien (centre négatif)
# Laplacian kernel 1
kernel_lap = np.array([
    [0,  1, 0],
    [1,  -4, 1],
    [0,  1, 0]
], dtype=np.float32)

# Laplacian kernel 2
kernel_lap1 = np.array([
    [0,  -1, 0],
    [-1, 4, -1],
    [0,  -1, 0]
], dtype=np.float32)

# Apply both kernels on Gaussian image
lap = cv2.filter2D(img_gauss, cv2.CV_32F, kernel_lap)
lap1 = cv2.filter2D(img_gauss, cv2.CV_32F, kernel_lap1)

# Sharpening with kernel 1
img_sharp = img_gris.astype(np.float32) + lap
img_sharp = np.clip(img_sharp, 0, 255).astype(np.uint8)

# Sharpening with kernel 2
img_sharp1 = img_gris.astype(np.float32) + lap1
img_sharp1 = np.clip(img_sharp1, 0, 255).astype(np.uint8)

# Display 3 images
cv2.imshow("image gauss", img_gauss)
cv2.imshow("Original Gray Image", img_gris)
cv2.imshow("Sharpening - Kernel 1", img_sharp)
cv2.imshow("Sharpening - Kernel 2", img_sharp1)

cv2.waitKey(0)
cv2.destroyAllWindows()

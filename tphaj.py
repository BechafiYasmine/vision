import cv2
import math
import numpy as np

img = cv2.imread("usthb.jpg")

if img is None:
    raise FileNotFoundError("image could not be loaded")

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

# Filtrage gaussien
img_gauss = cv2.filter2D(img_gris, cv2.CV_8U, kernel_gauss)

# Noyau Laplacien (centre négatif)
kernel_lap = np.array([
    [0,  -1, 0],
    [-1, 5, -1],
    [0,  -1, 0]
], dtype=np.float32)

# Laplacien sur l'image lissée (float pour garder les signes)
lap = cv2.filter2D(img_gauss, cv2.CV_32F, kernel_lap)


img_sharp = img_gris.astype(np.float32) #+lap
img_sharp = np.clip(img_sharp, 0, 255).astype(np.uint8)

# Pour l'affichage du Laplacien
lap_aff = cv2.convertScaleAbs(lap)

cv2.imshow("image grise", img_gris)
cv2.imshow("image gauss", img_gauss)
cv2.imshow("laplacien", lap_aff)
cv2.imshow("image sharp", img_sharp)
cv2.waitKey(0)
cv2.destroyAllWindows()

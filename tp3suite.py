import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL.ImageFilter import GaussianBlur

# Chargement de l'image en couleur
img = cv2.imread('img.jpeg', cv2.IMREAD_COLOR) # either we make it here imread_gray or we convert it later like we did here

if img is None:
    raise FileNotFoundError("Image could not be loaded !")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow("Gray", gray)
#
noise = np.random.normal(0, 25, gray.shape)
noisy = gray.astype(np.float32) + noise # to force calcule to be binaire 32
noisy = np.clip(noisy, 0, 255).astype(np.uint8) # clip = corriger les valeurs, without astype = nombre
# cv2.imshow("Noise", noise)
# cv2.imshow("Noisy", noisy)
#
# #********** Filter moyenne ******************************************************************
# # inconvenients: image flous , on perte les conteurs des objets
# def blur_image(value):
#     k = 2 * value + 1
#     mean_filtred = cv2.blur(noisy, (k, k)) #(k,k) selon le nombre de voisinage, moyenne simple
#     cv2.imshow("Blur", mean_filtred)
#
# cv2.namedWindow("Blur")
# cv2.createTrackbar("Kernel size", "Blur", 2, 20, blur_image)
# blur_image(2)
#
# #********** Filter gauss ******************************************************************
# # inconvenients: image flous , on perte les conteurs des objets, focus on central pixel more
# def gauss_blur_image(value):
#     k = 2 * value + 1
#     gaussian = cv2.GaussianBlur(noisy, (k, k) , 0) #moyenne pondere , les grands valeurs pour le centre ,
#     cv2.imshow("GaussianBlur", gaussian)
#
# cv2.namedWindow("GaussianBlur")
# cv2.createTrackbar("Kernel size", "GaussianBlur", 1, 20, gauss_blur_image)
# blur_image(2)
#
# #********** Filter Mediane ******************************************************************
# # inconvenients: image flous , focus more on edges
# def mediane_blur_image(value):
#     k = 2 * value + 1
#     mediane = cv2.medianBlur(noisy, k)
#     cv2.imshow("MedianeBlur", mediane)
#
# cv2.namedWindow("MedianeBlur")
# cv2.createTrackbar("Kernel size", "MedianeBlur", 1, 20, mediane_blur_image)
# blur_image(5)
#
# #********** Histogramme ******************************************************************
# hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
# plt.figure(figsize = (10,5))
#
# plt.plot(hist)
#
# plt.title("Image Histogram")
# plt.xlabel("Intensity")
# plt.ylabel("Frequenty")
# plt.show()
#
# #********** Histogramme pour rendre limage sombre ******************************************************************
# hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
# gray2 = (gray/2).astype(np.uint8)
# hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
#
# plt.figure(figsize = (10,5))
#
# plt.plot(hist)
# plt.plot(hist2)
#
# plt.title("Image Histogram")
# plt.xlabel("Intensity")
# plt.ylabel("Frequenty")
# plt.show()
#
# #********** Histogramme pour rendre limage sombre + claire ******************************************************************
# hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
# gray2 = (gray/2).astype(np.uint8)
# hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
# gray3 = (gray/2+127).astype(np.uint8)
# hist3 = cv2.calcHist([gray3], [0], None, [256], [0, 256])
#
# plt.figure(figsize = (10,5))
#
# plt.plot(hist)
# plt.plot(hist2)
# plt.plot(hist3)
#
# plt.title("Image Histogram")
# plt.xlabel("Intensity")
# plt.ylabel("Frequenty")
# plt.show()
#
# # Visualisation des images
# cv2.imshow("gray", gray)
# cv2.imshow("gray2", (gray/2).astype(np.uint8))
# cv2.imshow("gray3", (gray/2+127).astype(np.uint8))

#********** Histogramme Corriger les valeurs qui ne commence pas par 0 et le max pas forcement 255 ****************
# #(px - pmin) * 255 / (pmax - pmin)
# gray = (gray/2+50).astype(np.uint8)
# hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
# gray_equ = cv2.equalizeHist(gray) # pour fixer limage si sombre ou claire make it normal
# hist2 = cv2.calcHist([gray_equ], [0], None, [256], [0, 256])
#
# plt.figure(figsize = (10,5))
#
# plt.plot(hist)
# plt.plot(hist2)
#
# plt.title("Image Histogram")
# plt.xlabel("Intensity")
# plt.ylabel("Frequenty")
# plt.show()
#
# cv2.imshow("gray", gray)
# cv2.imshow("gray_equ", gray_equ)
#
# # por regler le probleme de decalage on peut appliquer le filter moyen

# #********** Histogramme equalisation + clahe ****************
# gray = (gray/2).astype(np.uint8)
#
# g_min, g_max = gray.min(), gray.max()
# gray_norm = ((gray.astype(np.float32) - g_min) / (g_max - g_min) * 255).astype(np.uint8)
#
# gray_equ = cv2.equalizeHist(gray) # pour fixer limage si sombre ou claire make it normal
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# gray_clahe = clahe.apply(gray)
#
# hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
# hist1 = cv2.calcHist([gray_norm], [0], None, [256], [0, 256])
# hist2 = cv2.calcHist([gray_equ], [0], None, [256], [0, 256])
# hist3 = cv2.calcHist([gray_clahe], [0], None, [256], [0, 256])
#
# plt.figure(figsize = (10,5))
#
# plt.plot(hist)
# plt.plot(hist1)
# plt.plot(hist2)
# plt.plot(hist3)
#
# plt.title("Image Histogram")
# plt.xlabel("Intensity")
# plt.ylabel("Frequenty")
# plt.show()
#
# cv2.imshow("gray_norm", gray)
# cv2.imshow("gray_equ", gray_equ)
# cv2.imshow("gray_clahe", gray_clahe)

#********** Gradient et Sobel ******************************************************************
th = 0
type_th = 0

def sobel(img, norm=True):
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0,ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx.astype(np.float32), gy.astype(np.float32))
    if not norm:
        return magnitude
    else:
        return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

sobel_img = sobel(img)
# cv2.imshow("sobel_img", sobel_img)

def gradient(img, norm = True):
    img = img.astype(np.float32)
    gradX = np.zeros_like(img)
    gradY = np.zeros_like(img)

    gradX[:,1:] = img[:,1:] - img[:,:-1]
    gradY[1:,:] = img[1:,:] - img[:-1,:]
    grad = np.sqrt(gradX**2 + gradY**2)

    if norm:
        grad = cv2.normalize(grad, None , 0 , 255 , cv2.NORM_MINMAX)
    else:
        grad = np.clip(grad, 0, 255)

    return grad.astype(np.uint8)

gradient_img = gradient(img)
# cv2.imshow("gradient_img", gradient_img)

def afficher():
    imgRes = np.zeros_like(img)
    cv2.threshold(img, th, 255, type_th, imgRes)
    cv2.imshow("img", imgRes)


def change_th(x):
    global th
    th = x
    afficher()


def change_type(x):
    global type_th
    type_th = x
    afficher()

afficher()

cv2.createTrackbar("thresh", "img", 0, 255, change_th)
cv2.createTrackbar("type", "img", 0, 4, change_type)



# # #********** Filter moyenne sans fonction predefinie ******************************************************************
# def filtreMoyen(img):
#
#     voisinage = 3
#     r = voisinage // 2
#
#     h, w = img.shape
#     imgMoy = np.zeros(img.shape, np.uint8)
#
#     for y in range(h):
#         for x in range(w):
#
#             # Check borders
#             if (x < r or x >= w-r or
#                 y < r or y >= h-r):
#                 imgMoy[y, x] = img[y, x]
#             else:
#                 # Extract the 3x3 neighborhood
#                 imgV = img[y-r:y+r+1, x-r:x+r+1]
#                 moy = 0
#                 # Calculate the sum
#                 for yv in range(voisinage):
#                     for xv in range(voisinage):
#                         moy += int(imgV[yv, xv])
#
#                 # Calculate the average
#                 moy = moy / (voisinage * voisinage)
#                 # Replace the pixel
#                 imgMoy[y, x] = int(moy)
#     return imgMoy
#
# imgMoy = filtreMoyen(noisy)
# # #********** Filter mediane sans fonction predefinie  ******************************************************************
# def filtreMediane(img):
#     voisinage = 3
#     r = voisinage // 2
#
#     h, w = img.shape
#     imgMed = np.zeros(img.shape, np.uint8)
#
#     for y in range(h):
#         for x in range(w):
#             # Check borders
#             if (x < r or x >= w-r or
#                 y < r or y >= h-r):
#                 imgMed[y, x] = img[y, x]
#             else:
#                 imgV = img[y - r:y + r + 1, x - r:x + r + 1]
#                 # Create an empty list
#                 t = np.zeros((voisinage*voisinage),np.uint8)
#                 # Extract the neighborhood
#                 for yv in range(voisinage):
#                     for xv in range(voisinage):
#                         t[yv*voisinage+xv] = imgV[yv,xv]
#
#                 # Sort manually using Bubble Sort
#                 for i in range(len(t)):
#                     for j in range(len(t)-1-i):
#                         if t[j] > t[j+1]:
#                             temp = t[j]
#                             t[j] = t[j+1]
#                             t[j+1] = temp
#
#                 # Retrieve the middle value
#                 imgMed[y, x] = t[int((voisinage * voisinage - 1) / 2)]
#
#     return imgMed
#
# imgMed = filtreMediane(noisy)
#
# # # ********** Histogram without predefined functions **********
# #
# # hist = np.zeros(256, np.uint32)
# #
# # h, w = gray.shape
# #
# # for y in range(h):
# #     for x in range(w):
# #         pixel = int(gray[y, x])
# #         hist[pixel] += 1
# #
# # plt.figure(figsize=(10, 5))
# # plt.plot(hist)
# #
# # plt.title("Original Image Histogram")
# # plt.xlabel("Intensity")
# # plt.ylabel("Frequency")
# # plt.grid()
# # plt.show()
#
#
# # ********** Image Normalization without predefined functions **********
#
# # Step 1: Create an empty result image
# imgNorm = np.zeros(noisy.shape, np.uint8)
#
# h, w = noisy.shape
# pmin = 255
# pmax = 0
#
# # Step 2: Find minimum and maximum intensity
# for y in range(h):
#     for x in range(w):
#
#         pixel = int(noisy[y, x])
#
#         if pixel < pmin:
#             pmin = pixel
#
#         if pixel > pmax:
#             pmax = pixel
#
#
# # Step 3: Normalize the image
# if pmax != pmin:
#
#     for y in range(h):
#         for x in range(w):
#
#             pixel = int(noisy[y, x])
#
#             imgNorm[y, x] = int(
#                 (pixel - pmin) * 255 / (pmax - pmin)
#             )
#
#
# # ********** Histograms before and after normalization **********
#
# hist1 = np.zeros(256, np.uint32)
# hist2 = np.zeros(256, np.uint32)
#
# h, w = noisy.shape
#
# for y in range(h):
#     for x in range(w):
#
#         pixel1 = int(noisy[y, x])
#         hist1[pixel1] += 1
#
#         pixel2 = int(imgNorm[y, x])
#         hist2[pixel2] += 1
#
#
# # Step 4: Display both histograms
# plt.figure(figsize=(10, 5))
#
# plt.plot(hist1, label="Before Normalization")
# plt.plot(hist2, label="After Normalization")
#
# plt.title("Histogram Comparison")
# plt.xlabel("Intensity")
# plt.ylabel("Frequency")
# plt.legend()
# plt.grid()
# plt.show()
#
#
# # ********** Display images **********
#
# cv2.imshow("Original", noisy)
# cv2.imshow("Normalized", imgNorm)
# cv2.imshow("Mean filter", imgMoy)
# cv2.imshow("Median filter", imgMed)















cv2.waitKey(0)          # Attend l'appui d'une touche
cv2.destroyAllWindows() # Ferme toutes les fenêtres OpenCV
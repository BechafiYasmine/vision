
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the image
img = cv2.imread("usthb.jpg", cv2.IMREAD_COLOR)

if img is None:
    raise FileNotFoundError("Image couldn't be loaded. Check the file path.")

# 2. Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3. Generate Gaussian noise
noise = np.random.normal(0, 25, gray.shape)

# 4. Add noise to the grayscale image
noisy = gray.astype(np.float32) + noise

# 5. Keep pixel values between 0 and 255
noisy = np.clip(noisy, 0, 255).astype(np.uint8)

# 6. Convert noise into a visible image for display
noise_display = cv2.normalize(
    noise, None, 0, 255, cv2.NORM_MINMAX
).astype(np.uint8)



# 7. Display the images
cv2.imshow("Gray Image", gray)
cv2.imshow("Noise", noise_display)
cv2.imshow("Noisy Image", noisy)

def blur_image(value):
    k=2 * value + 1
    
    mean_filtered = cv2.blur(noisy, (k, k))
    cv2.imshow("Blur", mean_filtered)
    
cv2.namedWindow("Blur")   
cv2.createTrackbar("kernel Size","Blur",2,20,blur_image)
blur_image(2)

def gauss_blur_image(value):
    k = 2 * value + 1
    gaussian = cv2.GaussianBlur(noisy, (k, k), 0)
    cv2.imshow("GaussianBlur", gaussian)

cv2.namedWindow("GaussianBlur")
cv2.createTrackbar("kernel Size", "GaussianBlur", 1, 20, gauss_blur_image)
gauss_blur_image(2)


def median(value):
    k = 2 * value + 1
    median_img = cv2.medianBlur(noisy, k)
    cv2.imshow("Median", median_img)

cv2.namedWindow("Median")
cv2.createTrackbar("kernel Size", "Median", 1, 20, median)
median(5)



#histogram 


#historam apres devision sur 2
hist=cv2.calcHist([gray], [0], None, [256], [0, 256])
gray2 = (gray / 2).astype(np.uint8)
gray3 = (gray / 2 + 127).astype(np.uint8)
hist2=cv2.calcHist([gray2], [0], None, [256], [0, 256])
hist3=cv2.calcHist([gray3], [0], None, [256], [0, 256])
plt.figure(figsize=(10,5))
plt.plot(hist)
plt.plot(hist2)
plt.plot(hist3)
plt.title("Histogramme de l'image")
plt.xlabel("Intensité des pixels")
plt.ylabel("Nombre de pixels")
plt.show()

#histogram apres egalisation

img = (gray * 255).astype(np.uint8)

hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

gray_eq = cv2.equalizeHist(img)

hist2 = cv2.calcHist([gray_eq], [0], None, [256], [0, 256])

plt.figure(figsize=(10, 5))
plt.plot(hist)
plt.plot(hist2)
plt.title("Image Histogram")
plt.xlabel("Intensity")
plt.ylabel("Frequency")
plt.show()
plt.xlabel("Intensité des pixels")
plt.ylabel("Nombre de pixels")
plt.show()

#********** Histogramme equalisation + clahe ****************
gray = (gray/2).astype(np.uint8)

g_min, g_max = gray.min(), gray.max()
gray_norm = ((gray.astype(np.float32) - g_min) / (g_max - g_min) * 255).astype(np.uint8)

gray_equ = cv2.equalizeHist(gray) # pour fixer limage si sombre ou claire make it normal
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray_clahe = clahe.apply(gray)

hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
hist1 = cv2.calcHist([gray_norm], [0], None, [256], [0, 256])
hist2 = cv2.calcHist([gray_equ], [0], None, [256], [0, 256])
hist3 = cv2.calcHist([gray_clahe], [0], None, [256], [0, 256])

plt.figure(figsize = (10,5))

plt.plot(hist)
plt.plot(hist1)
plt.plot(hist2)
plt.plot(hist3)

plt.title("Image Histogram")
plt.xlabel("Intensity")
plt.ylabel("Frequenty")
plt.show()

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
cv2.imshow("sobel_img", sobel_img)


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
cv2.imshow("gradient_img", gradient_img)


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



# Press a key in an OpenCV window to continue to the histogram.
cv2.waitKey(0)
cv2.destroyAllWindows()

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




# Press a key in an OpenCV window to continue to the histogram.
cv2.waitKey(0)
cv2.destroyAllWindows()
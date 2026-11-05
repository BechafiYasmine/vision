import cv2
import numpy as np

img = cv2.imread("usthb.jpg",cv2.IMREAD_COLOR)

if img is None:
    raise FileExistsError("image couldnt be loaded")



h,w,c = img.shape
img_res = np.zeros((h,w,c),dtype="uint8") #print(img-res[y,x]) --> [0 0 0]
#img_res = np.zeros((h,w),dtype="uint8") ca marche with or without the c :print(img-res[y,x]) --> 0

print("shape:",img.shape)
print("data type :",img.dtype)
print("number of dimensiosn",img.ndim)

for y in range(h):
    for x in range(w):
       # img_res[y,x,0] = 255 - img[y,x,0]
       # img_res[y,x,1] = 255 - img[y,x,1]
       # img_res[y,x,2] =255 - img[y,x,2]
       # img_res[y,x,:] = 255 -img[y,x,:]
       img_res[y,x] = (1/3*img[y,x,0]+1/3*img[y,x,1]+1/3*img[y,x,2])
       
#img_res[:,:] = 1/3*img[:,:,0]+1/3*img[:,:,1]+1/3*img[:,:,2]
img_res = np.uint8(1/3*img[:,:,0]+1/3*img[:,:,1]+1/3*img[:,:,2] )
img_res = (1/3*img[:,:,0]+1/3*img[:,:,1]+1/3*img[:,:,2] )/255 # tout les valeurs seront btw 1 et 

 #      img_res = 255-img
#img_res[:,:,:] = 255-img[:,:,:]
#img_res[:] = 255-img[:]




cv2.imshow('image1',img)
cv2.imshow('image_res',img_res)
cv2.imwrite("image_res.jpg",img_res)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.waitKey(0)
cv2.destroyWindow()   


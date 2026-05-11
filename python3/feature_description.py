import cv2
import numpy as np
from matplotlib import pyplot as plt



img = cv2.imread('../data/lena.jpg') # load a color image (default)
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# or
#gray = cv2.imread('../data/lena.jpg',0)



########### SIFT ###########


### Opencv 2.4.13
#sift = cv2.SIFT()
### Opencv 3
#sift = cv2.xfeatures2d.SIFT_create()
### Opencv 4
sift = cv2.SIFT_create()

kp, des = sift.detectAndCompute(gray,None)

img1 =cv2.drawKeypoints(gray, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

print("Number of SIFT descriptors : ", len(kp))
print("SIFT dimension : ", len(des[0]))

# display images with matplotlib
plt.imshow(img1), plt.title('SIFT')
plt.show()

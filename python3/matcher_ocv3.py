import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('../data/box.png',0)
img2 = cv2.imread('../data/box_in_scene.png',0) 


# Initiate SIFT detector
sift = cv2.SIFT_create()

# find the keypoints and descriptors with SIFT within the first image
kp1, des1 = sift.detectAndCompute(img1,None)
print("In img1 : ", len(des1), "descriptors \n")
img1kp=cv2.drawKeypoints(img1,kp1,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.figure(1)
plt.imshow(img1kp),plt.title('Keypoints of Image 1')


# Do the same for the second image
kp2, des2 = sift.detectAndCompute(img2,None)
print("In img2 : ", len(des2), "descriptors \n")
img2kp=cv2.drawKeypoints(img2,kp2,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.figure(2)
plt.imshow(img2kp),plt.title('Keypoints of Image 2')



# Search the k-nn of each descriptor if img1
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

print("Number of matches : ", len(matches))

# For the fun you can simply try this:
# matches = bf.knnMatch(des2, des1, k=2)

# Nicely presenting the results:
#
# Need to draw only good matches, so create a mask
# non matches are displayed but not linked.
matchesMask = [[0,0] for i in range(len(matches))]

nbMatches = 0
# Selection of good matches - TODO
for i,(m,n) in enumerate(matches):
    #m.distance = distance of the query desc queryIdx to its 1-nn (trainIdx)
    #n.distance = distance of the query desc queryIdx to its 2-nn (trainIdx)
    matchesMask[i]=[1,0]
    nbMatches+=1


draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

print("There are ", nbMatches, " matches according to the matching criteria\n")
plt.figure(3)
plt.imshow(img3),plt.title('Matches')
plt.show()


# 1. Compare different matching criteria
# 2. Compare number of matches between similar images and dissimilar images
# 3. Try with different images: textured, logos, faces...
# 4. Using Gimp and/or opencv, produce attacked versions of an image: blur, crop, rotation, jpeg compression with various quality factors, occultations, flip, noise
# 5. Check which transformations are the most difficult for correctly matching descriptors

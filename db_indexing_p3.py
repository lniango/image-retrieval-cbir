import os
import sys
import glob
import cv2
import numpy as np
import argparse

from timeit import default_timer as timer

''' 
    Usage :
    ./db_indexing_p3.py -d "database_name"
    
    Example :
    ./db_indexing_p3.py -d "base1"
'''

######## Program parameters

parser = argparse.ArgumentParser()

## Database name
parser.add_argument("-d", "--database", dest="db_name",
                    help="input image database", metavar="STRING",default="None")

args = parser.parse_args()

## Set paths [TO UPDATE]
img_dir="C:/Users/HP/Documents/ESIR Formation/Semestre 9/AMM/Projet/AMM/Images/" + args.db_name + "/"
imagesNameList = glob.glob(img_dir+"*.jpg")
output_dir="./databases/" + args.db_name

if not os.path.exists(img_dir):
    print("The directory containing images: "+img_dir+" is not found -- EXIT\n")
    sys.exit(1)

####################
#### Compute descriptors of the whole database
####################
dataBaseDescriptors = []
imageBaseIndex = []

im_nb = 0
des_nb=0


### Opencv 4
sift = cv2.SIFT_create()


for imageName in imagesNameList: #[:500]:
    print(str(im_nb+1) + " Compute descriptors for : " + imageName)

    image = cv2.imread(imageName)
    kp, des = sift.detectAndCompute(image,None)

    if(des is not None):
        for descriptor in des:
            dataBaseDescriptors.append(descriptor)
            imageBaseIndex.append(im_nb)
            des_nb+=1

    im_nb = im_nb + 1

print(str(im_nb) + " images in the DB")
print(str(des_nb) + " descriptors in the DB")

np.save(output_dir + "_DB_Descriptors.npy",dataBaseDescriptors)  #stores all the SIFT for all images
np.save(output_dir + "_imagesIndex.npy",imageBaseIndex) # gives a number to each image
np.save(output_dir + "_imagesPaths.npy",imagesNameList) # gives the path to each image



####################
### Index database descriptors
####################

## Load descriptors (if not in memory)
dataBaseDescriptors = np.load(output_dir + "_DB_Descriptors.npy")


# Algorithms
# 0 : FLANN_INDEX_LINEAR,
# 1 : FLANN_INDEX_KDTREE,

# No index. Subsequent sequential and exhaustive scan.
FLANN_INDEX_ALGO=0
print("Creating LINEAR index")
start = timer()
index_params = dict(algorithm = FLANN_INDEX_ALGO)
### Opencv 4
fl = cv2.flann_Index(np.asarray(dataBaseDescriptors,np.float32),index_params)
end = timer()
print("linear index created in: " + str(end - start))
fl.save(output_dir + "_flann_index-LINEAR.dat")

# KDtree index
FLANN_INDEX_ALGO=1
print("\n\nCreating KDTREE index")
start = timer()
index_params = dict(algorithm = FLANN_INDEX_ALGO, trees = 5)
### Opencv 4
fl = cv2.flann_Index(np.asarray(dataBaseDescriptors,np.float32),index_params)
end = timer()
print("kdtree index descriptors: " + str(end - start))
fl.save(output_dir + "_flann_index-KDTREE.dat")


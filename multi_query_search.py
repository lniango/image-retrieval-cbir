import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import sys
from timeit import default_timer as timer

'''
    Usage:
    ./query_search_p3.py -d "database_name" -q "query_imagename" -t "index_type" -r "relevant_images_number"
    
    Example:
    python query_search_p3.py -d COREL -q corel_0000000303_512 -t LINEAR
'''


######## Program parameters

import argparse
parser = argparse.ArgumentParser()

## Database name
parser.add_argument("-d", "--database", dest="db_name",
                    help="input image database", metavar="STRING")


## Query image name
parser.add_argument("-q", "--query", dest="query_name",
                    help="query image name", metavar="STRING")

## Database Index Type
parser.add_argument("-t", "--indextype", dest="indextype",
                    help="index type", metavar="STRING")

## Number of relevant images in the database, considering the query
parser.add_argument("-r", "--relevant", dest="relevant",  type=int,
                    help="relevant image number", metavar="INTEGER", default=4)

args = parser.parse_args()


## Set paths
img_dir="C:/Users/HP/Documents/ESIR Formation/Semestre 9/AMM/Projet/AMM/Images/" + args.db_name + "/"
if (args.db_name == "COREL" or args.db_name == "NISTER" or args.db_name == "Copydays"):
    img_dir="C:/Users/HP/Documents/ESIR Formation/Semestre 9/AMM/Projet/AMM/Images/" + args.db_name + "_queries/"
db_dir="./databases/"+ args.db_name
resfilename = "./results/" + args.query_name + "-" + args.indextype


## Load query image
query_filename=img_dir + args.query_name + ".jpg"
if not os.path.isfile(query_filename):
    print("Path to the query "+query_filename+" is not found -- EXIT\n")
    sys.exit(1)

queryImage = cv2.imread(query_filename)

plt.figure(0), plt.title("Image requete")
plt.imshow(cv2.cvtColor(queryImage, cv2.COLOR_BGR2RGB))


## Compute query descriptors
### Opencv 4
sift = cv2.SIFT_create()

kp, qdesc = sift.detectAndCompute(queryImage,None)
print("Number of query descriptors :", len(qdesc))

#######################
## Search for similar descriptors in the database
# 1/ load the descriptors for final distance calculations
#    load the data to map image numbers to image names for final display
# 2/ load index and run K-NN query
#
#######################

## Load database descriptors **this is not the index**
start = timer()
dataBaseDescriptors = np.load(db_dir + "_DB_Descriptors.npy")
imageBasePaths = np.load(db_dir +"_imagesPaths.npy")
imageBaseIndex = np.load(db_dir +"_imagesIndex.npy")
end = timer()
print("time to load descriptors: " + str(end - start))



## Load database index (computed offline)
# algorithm = 254 is the parameter to use in order to LOAD AN EXISTING INDEX !!!
start = timer() 
index_params = dict(algorithm = 254, filename = db_dir +"_flann_index-" + args.indextype +".dat")
### Opencv 4.5
fl = cv2.flann_Index(np.asarray(dataBaseDescriptors,np.float32),index_params)
end = timer()
print("time to load the index: " + str(end - start))



def getImageId(imname):
    if (args.db_name == "COREL"):
        Id = imname.split('_')[1]
    elif (args.db_name == "NISTER"):
        Id = imname.split('-')[1]
    elif (args.db_name == "Copydays"):
        Id = imname.split('_')[-2]
    else:
        Id = imname.split('.')[-1]
    
    return Id






query_images = [f.split('.')[0] for f in os.listdir(img_dir) if f.endswith('.jpg')]
i=0

# Variables pour stocker les métriques globales
all_precisions = []
all_recalls = []
all_ap = []

for query in query_images:
    query_filename = img_dir + query + ".jpg"
    queryImage=cv2.imread(query_filename)

    sift = cv2.SIFT_create()

    kp, qdesc = sift.detectAndCompute(queryImage,None)
    print("Number of query descriptors :", len(qdesc))

    knn = 5
    idx, dist = fl.knnSearch(np.asarray(qdesc, np.float32), knn, params={})

    image_scores = np.zeros(len(imageBasePaths), dtype=np.int32)


    for i in range(idx.shape[0]):  
        for j in range(knn):  
            db_image_index = imageBaseIndex[idx[i, j]]  
            image_scores[db_image_index] += 1  


    filtered_scores = []
    for i, score in enumerate(image_scores):
        if score > 0:  
            filtered_scores.append((score, imageBasePaths[i]))


    filtered_scores = sorted(filtered_scores, key=lambda x: x[0], reverse=True)
    top=10
    plt.figure(1), plt.title(args.indextype )
    for i in range(top):
        img = cv2.imread(filtered_scores[i][1])
        score = filtered_scores[i][0]
        plt.subplot(2,5,i+1),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('rank '+str(i+1)), plt.xticks([]), plt.yticks([]),plt.xlabel(str(score))

    plt.savefig(resfilename + "_top" + str(top) +".png")
    #plt.show()

    queryId = getImageId(query)
    precision = np.zeros(len(filtered_scores), dtype=float)
    recall = np.zeros(len(filtered_scores), dtype=float)

    true_positif = 0
    num_ap = 0.0

    for i, (score, path) in enumerate(filtered_scores):
        if getImageId(path) == queryId:
            true_positif += 1
            num_ap += true_positif / (i + 1)

        precision[i] = true_positif / (i + 1)
        recall[i] = true_positif / args.relevant

    ap = num_ap / true_positif if true_positif > 0 else 0
    all_precisions.append(precision)
    all_recalls.append(recall)
    all_ap.append(ap)

mean_precision = np.mean([np.mean(p) for p in all_precisions]) if all_precisions else 0
mean_recall = np.mean([np.mean(r) for r in all_recalls]) if all_recalls else 0
mAP = np.mean(all_ap) if all_ap else 0

# Affichage des résultats globaux
print("Précision moyenne:", mean_precision)
print("Rappel moyen", mean_recall)
print("mAP",mAP)

# Affichage de la courbe PR globale
plt.figure()
for recall, precision in zip(all_recalls, all_precisions):
    plt.plot(recall, precision, lw=1, alpha=0.5)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Global)')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.05])
plt.grid(True)
plt.show()

        
        
    
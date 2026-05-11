import cv2
import numpy as np
from timeit import default_timer as timer

####### Utilisation flann
# - les matrices descripteurs ou requetes doivent etre du type 'np.float32'
# - le vecteur requete doit etre correctement formate : qdesc.shape = (1, desc_dim)
# -> reshape
# distance (a,b) calculee par knnSearch = sum (ai-bi)^2
########


#### Generer aleatoirement une matrice des descripteurs de la base
# - randint(100) : distribution uniforme, nombres entiers entre 0 et 100
# - size=(10,5) : 10 descripteurs de dimension 5
dim=5
db_size=10
mat_desc=np.array(np.random.randint(100,size=(db_size,dim)),dtype=np.float32)
print("Database descriptors : \n", mat_desc)


#### Generer un vecteur ou une matrice de vecteurs requete
q_size=1
qdesc=np.array(np.random.randint(100,size=(q_size,dim)),dtype=np.float32)
print("Query descriptor : \n", qdesc)



######## Database Descriptors indexing
# FLANN parameters
# Algorithms
# 0 : FLANN_INDEX_LINEAR,
# 1 : FLANN_INDEX_KDTREE,


start = timer()
FLANN_INDEX_ALGO=0
index_params = dict(algorithm = FLANN_INDEX_ALGO)   # for linear search
#index_params = dict(algorithm = FLANN_INDEX_ALGO, trees = 5) # for kdtree search

### OpenCV 3
fl=cv2.flann_Index(mat_desc,index_params)

end = timer()
print("Indexing time: " + str(end - start))


######## Query search
start = timer()
knn=5
#search_params = dict(checks=50)
idx, dist=fl.knnSearch(qdesc,knn,params={})
end = timer()
print("Search time: " + str(end - start))


#print idx.shape
print("indices \n", idx)
print("distances \n", dist)

#Thresholding the distance (Radius Search)
#print idx[dist<3000]



# 1. test the search with a set of query descriptors
# 2. change the descriptors dimension (d=500) and the number of DB descriptors (N=1000)
# 3. modify the DB size = 10 000, 100 0000, and 1 000 000 (change knn values accordingly)
# 4. compare the computation time (indexing and search) in each case for both linear search and KDT tree

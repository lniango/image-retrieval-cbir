
import os
import glob
import numpy as np
import argparse
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--database", dest="db_name", help="Nom de la base d'images", metavar="STRING", default="None")
args = parser.parse_args()

# Configurer les chemins
img_dir = "C:/Users/HP/Mettre ton chemin/Images/" + args.db_name + "/"  # Dossier contenant les images
output_dir = "C:/Users/HP/Mettre ton chemin/databasesKNN" + args.db_name  # Sauvegarder les descripteurs
os.makedirs(output_dir, exist_ok=True)  # Créer le dossier si nécessaire

# Charger le modèle VGG16 (avec pooling global average)
model = VGG16(weights='imagenet', include_top=False, pooling='avg')

# Liste des images
imagesNameList = glob.glob(img_dir + "*.jpg")
dataBaseDescriptors = []  # Liste pour stocker les descripteurs globaux
imagePaths = []  # Liste pour stocker les chemins des images

# Traitement de chaque image
for i, imageName in enumerate(imagesNameList):
    print(f"({i+1}/{len(imagesNameList)}) Traitement de : {imageName}")
    img = load_img(imageName, target_size=(224, 224))  # Charger l'image et redimensionner
    x = img_to_array(img)  # Convertir en tableau NumPy
    x = np.expand_dims(x, axis=0)  # Ajouter une dimension batch
    x = preprocess_input(x)  # Prétraitement VGG16
    features = model.predict(x)  # Extraire les descripteurs globaux
    dataBaseDescriptors.append(features.flatten())  # Ajouter les descripteurs aplatis
    imagePaths.append(imageName)  # Ajouter le chemin de l'image

# Sauvegarder les descripteurs et les chemins des images
np.save(output_dir + "_DB_Descriptors.npy", np.array(dataBaseDescriptors))  # Descripteurs globaux
np.save(output_dir + "_imagesPaths.npy", np.array(imagePaths))  # Chemins des images

print(f"Indexation terminée pour {len(imagesNameList)} images. Fichiers sauvegardés dans : {output_dir}")
# export PYTHONPATH="/usr/local/lib/python3.8/dist-packages"
# see: https://keras.io/applications/

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
import numpy as np

base_model = VGG19(weights='imagenet')

model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)
#model.summary()

img_path = '../data/lena.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)

# our image is now represented by a NumPy array of shape (3, 224, 224),
# but we need to expand the dimensions to be (1, 3, 224, 224) so we can
# pass it through the network.
# This is related to batch learning
x = np.expand_dims(x, axis=0)
x = preprocess_input(x) # Adequate the image to the format the model requires (for ex. subtracting the mean RGB pixel intensity from the ImageNet dataset)

block4_pool_features = model.predict(x)
print("last convolutionnal layer size : ", block4_pool_features.shape)
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
import numpy as np
# Load pre-trained VGG16 model
model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Specify the path to the folder containing images
folder_path = './Dataset/RoughBark/IMG_4844.jpg'

# Iterate over each image in the folder
# for filename in os.listdir(folder_path):
    # if filename.endswith(".jpg") or filename.endswith(".png"):  # Assuming images are in jpg or png format
        # Load and preprocess the image
        # img_path = os.path.join(folder_path, filename)
img = image.load_img(folder_path, target_size=(224, 224))
x = image.img_to_array(img)
x = preprocess_input(x)
x = np.expand_dims(x, axis=0)

# Get model predictions
predictions = model.predict(x)

features_flattened = predictions.flatten()
print(features_flattened)

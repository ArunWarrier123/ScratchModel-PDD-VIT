import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os
from vit import ViT
import cv2

""" Hyperparameters """
hp = {}
hp["image_size"] = 128
hp["num_channels"] = 3
hp["patch_size"] = 16
hp["num_patches"] = (hp["image_size"]**2) // (hp["patch_size"]**2)
hp["flat_patches_shape"] = (hp["num_patches"], hp["patch_size"]*hp["patch_size"]*hp["num_channels"])

hp["batch_size"] = 16
hp["lr"] = 1e-4
hp["num_epochs"] = 20
hp["num_classes"] = 2
hp["class_names"] = ["RoughBark","StripeCanker"]

hp["num_layers"] = 12
hp["hidden_dim"] = 384
hp["mlp_dim"] = 1536
hp["num_heads"] = 6
hp["dropout_rate"] = 0.2
# Load the model
# model = load_model('your_model.h5')
model_path = os.path.join("files", "modelpatch_16.h5")

""" Model """
model = ViT(hp)
model.load_weights(model_path)
model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
    optimizer=tf.keras.optimizers.Adam(hp["lr"]),
    metrics=["acc"]
)

# model.summary()
# # Identify the last convolutional layer
last_conv_layer = model.get_layer('dense_25')

# # Create a new model for CAM
cam_model = tf.keras.Model(model.input, outputs=[last_conv_layer.output, model.output])


# cam_model.summary()
# Load and preprocess misclassified images
img_path = './Dataset/RoughBark/IMG_4844.jpg'
img = image.load_img(img_path, target_size=(64, 768))  # Adjust target_size as needed
img_array = image.img_to_array(img)
# img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Normalize the image

# # Get the class activation map
last_conv_output = cam_model.predict(img_array)
print(last_conv_output)
# # class_index = np.argmax(preds[0])
# # cam = np.dot(last_conv_output[0, ..., class_index], model.get_layer('dense_25').get_weights()[0])

# # # Resize the CAM to match the original image size
# # cam = tf.image.resize(cam, (img_array.shape[1], img_array.shape[2]))

# # # Normalize the CAM
# # cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))

# # # Overlay the CAM on the original image
# # heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
# # superimposed_img = cv2.addWeighted(cv2.cvtColor(img_array[0], cv2.COLOR_RGB2BGR), 0.5, heatmap, 0.5, 0)

# # # Display the original image, CAM, and the superimposed image
# # plt.imshow(img)
# # plt.title('Original Image')
# # plt.show()

# # plt.imshow(cam, cmap='jet')
# # plt.title('Class Activation Map')
# # plt.show()

# # plt.imshow(superimposed_img)
# # plt.title('Superimposed Image with CAM')
# # plt.show()

import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from vit import ViT


""" Hyperparameters """
hp = {}
hp["image_size"] = 200
hp["num_channels"] = 3
hp["patch_size"] = 25
hp["num_patches"] = (hp["image_size"]**2) // (hp["patch_size"]**2)
hp["flat_patches_shape"] = (hp["num_patches"], hp["patch_size"]*hp["patch_size"]*hp["num_channels"])

hp["batch_size"] = 32
hp["lr"] = 1e-4
hp["num_epochs"] = 1
hp["num_classes"] = 2
hp["class_names"] = ["RoughBark","StripeCanker"]

hp["num_layers"] = 12
hp["hidden_dim"] = 768
hp["mlp_dim"] = 3072
hp["num_heads"] = 12
hp["dropout_rate"] = 0.1

model = ViT(hp)
model.load_weights("./files/model.h5")
model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        optimizer=tf.keras.optimizers.Adam(hp["lr"]),
        metrics=["acc"]
    )
# Load and preprocess your image
image_path = './Dataset/RoughBark/IMG_4844.jpg'
image = tf.keras.preprocessing.image.load_img(image_path, target_size=(64, 1875))
image_array = tf.keras.preprocessing.image.img_to_array(image)
image_array = tf.expand_dims(image_array, 0)  # Add batch dimension
image_array /= 255.0  # Normalize pixel values to [0, 1]

# Create a submodel to get the attention maps
attention_map_model = tf.keras.Model(
    inputs=model.input,
    outputs=[layer.output for layer in model.layers if 'multi_head_attention' in layer.name]
)

# Get the attention maps for a specific layer and head
layer_idx = 2  # Replace with the index of the desired multi_head_attention layer
head_idx = 0   # Replace with the index of the desired head

attention_maps = attention_map_model.predict(image_array)
attention_map = attention_maps[layer_idx][0][:, :, head_idx]

# Resize attention map to match the input image size
attention_map_resized = tf.image.resize(attention_map, (64, 1875)).numpy()

# Plot the original image
plt.figure(figsize=(12, 6))
plt.imshow(image)
plt.axis('off')
plt.title('Original Image')

# Plot the attention map
plt.figure(figsize=(12, 6))
sns.heatmap(attention_map_resized, cmap='viridis', square=True, cbar=False)
plt.title(f'Attention Map\nLayer: {layer_idx}, Head: {head_idx}')
plt.show()

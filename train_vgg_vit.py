
# import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# import numpy as np
# import cv2
# from glob import glob
# from sklearn.utils import shuffle
# from sklearn.model_selection import train_test_split
# from patchify import patchify
# import tensorflow as tf
# from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
# from vit import ViT

# """ Hyperparameters """
# hp = {}
# hp["image_size"] = 128 #200 to 128
# hp["num_channels"] = 3
# hp["patch_size"] = 16 # 25 to 16
# hp["num_patches"] = (hp["image_size"]**2) // (hp["patch_size"]**2)
# hp["flat_patches_shape"] = (hp["num_patches"], hp["patch_size"]*hp["patch_size"]*hp["num_channels"])

# hp["batch_size"] = 16 # 32 to 16
# hp["lr"] = 1e-4
# hp["num_epochs"] = 20 # 30 to 20
# hp["num_classes"] = 2
# hp["class_names"] = ["RoughBark","StripeCanker"]

# hp["num_layers"] = 12
# hp["hidden_dim"] = 384 # 768 to 384
# hp["mlp_dim"] = 1536 #3072 to 1536
# hp["num_heads"] = 6 # 12 to 6
# hp["dropout_rate"] = 0.2 #0.1 to 0.2

# def create_dir(path):
#     if not os.path.exists(path):
#         os.makedirs(path)

# def load_data(path, split=0.1):
#     images = shuffle(glob(os.path.join(path, "*", "*.jpg")))
#     # print(path)
#     split_size = int(len(images) * split)
#     train_x, valid_x = train_test_split(images, test_size=split_size, random_state=42)
#     train_x, test_x = train_test_split(train_x, test_size=split_size, random_state=42)

#     return train_x, valid_x, test_x

# def process_image_label(path):
#     """ Reading images """
#     path = path.decode()
#     image = cv2.imread(path, cv2.IMREAD_COLOR)
#     image = cv2.resize(image, (hp["image_size"], hp["image_size"]))
#     image = image/255.0

#     """ Preprocessing to patches """
#     patch_shape = (hp["patch_size"], hp["patch_size"], hp["num_channels"])
#     patches = patchify(image, patch_shape, hp["patch_size"])

#     # patches = np.reshape(patches, (64, 25, 25, 3))
#     # for i in range(64):
#     #     cv2.imwrite(f"files/{i}.png", patches[i])

#     # old 2 lines of code below
#     patches = np.reshape(patches, hp["flat_patches_shape"])
#     patches = patches.astype(np.float32)

#     #new 2 lines of code below to change hp patch size
#     # patches = np.reshape(patches, (-1, hp["patch_size"] * hp["patch_size"] * hp["num_channels"]))
#     # patches = patches.astype(np.float32)

#     """ Label """
#     # print({"path": path})
#     # class_name = path.split("/")[-2]
#     # print({"classname": class_name})
#     normalized_path = os.path.normpath(path)

#     # Split the normalized path
#     path_parts = normalized_path.split(os.sep)

#     # Extract the second-to-last part as the class name
#     class_name = path_parts[-2]

#     # print(class_name)
#     class_idx = hp["class_names"].index(class_name)
#     class_idx = np.array(class_idx, dtype=np.int32)

#     return patches, class_idx

# def parse(path):
#     patches, labels = tf.numpy_function(process_image_label, [path], [tf.float32, tf.int32])
#     labels = tf.one_hot(labels, hp["num_classes"])

#     patches.set_shape(hp["flat_patches_shape"])
#     labels.set_shape(hp["num_classes"])

#     # the below image_name use only for running test.py
#     # Extract the image name from the path
#     image_name = tf.strings.split(path, os.path.sep)[-1]
#     return patches, labels , image_name

# def tf_dataset(images, batch=32):
#     ds = tf.data.Dataset.from_tensor_slices((images))
#     ds = ds.map(parse).batch(batch).prefetch(8)
#     return ds


# if __name__ == "__main__":
#     """ Seeding """
#     np.random.seed(42)
#     tf.random.set_seed(42)

#     """ Directory for storing files """
#     create_dir("files")

#     """ Paths """
#     dataset_path = "./Dataset"
#     model_path = os.path.join("files", "modelpatchvgg_16.h5")
#     csv_path = os.path.join("files", "logpatchvgg_16.csv")

#     """ Dataset """
#     train_x, valid_x, test_x = load_data(dataset_path)
#     print(f"Train: {len(train_x)} - Valid: {len(valid_x)} - Test: {len(test_x)}")

#     train_ds = tf_dataset(train_x, batch=hp["batch_size"])
#     valid_ds = tf_dataset(valid_x, batch=hp["batch_size"])

#     """ Model """
#     model = ViT(hp)
#     model.compile(
#         loss="categorical_crossentropy",
#         optimizer=tf.keras.optimizers.Adam(hp["lr"], clipvalue=1.0),
#         metrics=["acc"]
#     )

#     callbacks = [
#         ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True),
#         ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-10, verbose=1),
#         CSVLogger(csv_path),
#         EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=False),
#     ]
    
#     history = model.fit(
#         train_ds,
#         epochs=hp["num_epochs"],
#         validation_data=valid_ds,
#         callbacks=callbacks
#     )

#     import pickle
#     with open('training_history_patch_16.pkl', 'wb') as file:
#         pickle.dump(history.history, file)

#     ## ...



import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
from glob import glob
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from patchify import patchify
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
# from vit_vgg16 import ViT
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Input, Flatten, Reshape, Dense, Embedding, Concatenate


""" Hyperparameters """
hp = {}
hp["image_size"] = 200
hp["num_channels"] = 3
hp["patch_size"] = 25
hp["num_patches"] = (hp["image_size"]**2) // (hp["patch_size"]**2)
hp["flat_patches_shape"] = (hp["num_patches"], hp["patch_size"]*hp["patch_size"]*hp["num_channels"])

hp["batch_size"] = 32
hp["lr"] = 1e-4
hp["num_epochs"] = 30
hp["num_classes"] = 2
hp["class_names"] = ["RoughBark","StripeCanker"]

hp["num_layers"] = 12
hp["hidden_dim"] = 768
hp["mlp_dim"] = 3072
hp["num_heads"] = 12
hp["dropout_rate"] = 0.1


def ViT_with_VGG16(cf):
    """ Inputs """
    vgg16_input_shape = (224, 224, 3)

    # Load pre-trained VGG16 model
    vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=vgg16_input_shape)

    """ VGG16 as Feature Extractor """
    vgg16_input = Input(shape=vgg16_input_shape)
    vgg16_features = vgg16_model(vgg16_input)
    vgg16_flattened = Flatten()(vgg16_features)

    """ ViT Architecture """
    input_shape = (cf["num_patches"], cf["hidden_dim"] + vgg16_flattened.shape[-1])
    inputs = Input(input_shape)

    # Modify ViT input by concatenating VGG16 features
    patch_embed = Dense(cf["hidden_dim"])(inputs)
    positions = tf.range(start=0, limit=cf["num_patches"], delta=1)
    pos_embed = Embedding(input_dim=cf["num_patches"], output_dim=cf["hidden_dim"])(positions)
    embed = patch_embed + pos_embed

    # Concatenate VGG16 features and ViT input
    # vgg16_flattened = Reshape((1, vgg16_flattened.shape[-1]))(vgg16_flattened)
    # combined_input = Concatenate(axis=1)([vgg16_flattened, embed])
    
    vgg16_flattened = Flatten()(vgg16_features)
    vgg16_flattened = Reshape((1, -1))(vgg16_flattened)
    combined_input = Concatenate(axis=1)([vgg16_flattened, embed])
    # Apply ViT transformer blocks
    for _ in range(cf["num_layers"]):
        combined_input = transformer_encoder(combined_input, cf)

    """ Classification Head """
    combined_input = LayerNormalization()(combined_input)
    combined_input = combined_input[:, 0, :]
    x = Dense(cf["num_classes"], activation="softmax")(combined_input)

    """ Combine VGG16 and ViT models """
    combined_model = Model(inputs=[vgg16_input, inputs], outputs=x)
    return combined_model


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path, split=0.1):
    images = shuffle(glob(os.path.join(path, "*", "*.jpg")))
    # print(path)
    split_size = int(len(images) * split)
    train_x, valid_x = train_test_split(images, test_size=split_size, random_state=42)
    train_x, test_x = train_test_split(train_x, test_size=split_size, random_state=42)

    return train_x, valid_x, test_x

def process_image_label(path):
    """ Reading images """
    path = path.decode()
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (hp["image_size"], hp["image_size"]))
    image = image/255.0

    """ Preprocessing to patches """
    patch_shape = (hp["patch_size"], hp["patch_size"], hp["num_channels"])
    patches = patchify(image, patch_shape, hp["patch_size"])

    # patches = np.reshape(patches, (64, 25, 25, 3))
    # for i in range(64):
    #     cv2.imwrite(f"files/{i}.png", patches[i])

    patches = np.reshape(patches, hp["flat_patches_shape"])
    patches = patches.astype(np.float32)

    """ Label """
    # print({"path": path})
    # class_name = path.split("/")[-2]
    # print({"classname": class_name})
    normalized_path = os.path.normpath(path)

    # Split the normalized path
    path_parts = normalized_path.split(os.sep)

    # Extract the second-to-last part as the class name
    class_name = path_parts[-2]

    # print(class_name)
    class_idx = hp["class_names"].index(class_name)
    class_idx = np.array(class_idx, dtype=np.int32)

    return patches, class_idx

def parse(path):
    patches, labels = tf.numpy_function(process_image_label, [path], [tf.float32, tf.int32])
    labels = tf.one_hot(labels, hp["num_classes"])

    patches.set_shape(hp["flat_patches_shape"])
    labels.set_shape(hp["num_classes"])

    return patches, labels

def tf_dataset(images, batch=32):
    ds = tf.data.Dataset.from_tensor_slices((images))
    ds = ds.map(parse).batch(batch).prefetch(8)
    return ds


if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("files")

    """ Paths """
    dataset_path = "./Dataset"
    model_path = os.path.join("files", "model.h5")
    csv_path = os.path.join("files", "log.csv")

    """ Dataset """
    train_x, valid_x, test_x = load_data(dataset_path)
    print(f"Train: {len(train_x)} - Valid: {len(valid_x)} - Test: {len(test_x)}")

    train_ds = tf_dataset(train_x, batch=hp["batch_size"])
    valid_ds = tf_dataset(valid_x, batch=hp["batch_size"])

    """ Model """
    model = ViT_with_VGG16(hp)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(hp["lr"], clipvalue=1.0),
        metrics=["acc"]
    )

    callbacks = [
        ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-10, verbose=1),
        CSVLogger(csv_path),
        EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=False),
    ]
    print(train_ds)
    history = model.fit(
        train_ds,
        epochs=hp["num_epochs"],
        validation_data=valid_ds,
        callbacks=callbacks
    )

    import pickle
    with open('training_history.pkl', 'wb') as file:
        pickle.dump(history.history, file)

    ## ...

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
from patchify import patchify
import tensorflow as tf
from train import load_data, tf_dataset
from vit import ViT
from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt

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


if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Paths """
    dataset_path = "./Dataset"
    model_path = os.path.join("files", "modelpatch_16.h5")

    """ Dataset """
    train_x, valid_x, test_x = load_data(dataset_path)
    print(f"Train: {len(train_x)} - Valid: {len(valid_x)} - Test: {len(test_x)}")

    test_ds = tf_dataset(test_x, batch=hp["batch_size"])

    """ Model """
    model = ViT(hp)
    model.load_weights(model_path)
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        optimizer=tf.keras.optimizers.Adam(hp["lr"]),
        metrics=["acc"]
    )

    eval_results =  model.evaluate(test_ds)
    print("Loss:", eval_results[0])
    print("Accuracy:", eval_results[1])

    # Evaluate the model on the test set
    test_predictions = model.predict(test_ds)
    test_true_labels = np.concatenate([labels.numpy() for patches, labels in test_ds])

    # Convert one-hot encoded labels back to class indices
    test_true_indices = np.argmax(test_true_labels, axis=1)
    test_predicted_indices = np.argmax(test_predictions, axis=1)

    # Generate confusion matrix
    conf_matrix = confusion_matrix(test_true_indices, test_predicted_indices)

    # Print the confusion matrix
    print("Confusion Matrix:")
    print(conf_matrix)

    # Print classification report
    class_names = hp["class_names"]
    print("\nClassification Report:")
    print(classification_report(test_true_indices, test_predicted_indices, target_names=class_names))


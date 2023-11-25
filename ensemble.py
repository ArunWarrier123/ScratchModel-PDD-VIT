import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from sklearn.metrics import accuracy_score
import numpy as np
import cv2
import seaborn as sns
from patchify import patchify
import tensorflow as tf
from train import load_data, tf_dataset
from vit import ViT
from sklearn.metrics import confusion_matrix, classification_report , roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score 
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions


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


if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Paths """
    dataset_path = "./Dataset"
    model_path = os.path.join("files", "model_ensemble.h5")

    """ Dataset """
    train_x, valid_x, test_x = load_data(dataset_path)
    print(f"Train: {len(train_x)} - Valid: {len(valid_x)} - Test: {len(test_x)}")
    all_x = []
    all_x.extend(train_x)
    all_x.extend(valid_x)
    all_x.extend(test_x)

    test_ds = tf_dataset(all_x, batch=hp["batch_size"])

    """ Model """
    model = ViT(hp)
    model.load_weights(model_path)
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        optimizer=tf.keras.optimizers.Adam(hp["lr"]),
        metrics=["acc"]
    )

     # Evaluate the model on the test set
    vit_predictions = model.predict(test_ds)



    #VGG TEST
    # Load the VGG model
    vgg_model_path = 'vgg16_EP-20_CP.h5'
    vgg_model = load_model(vgg_model_path)

    # Define the path to your dataset
    dataset_path = './Dataset'

    # Define parameters
    img_width, img_height = 224, 224
    batch_size = 32

    # Create a data generator for the dataset
    datagen = image.ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    # Load and preprocess images for prediction
    generator = datagen.flow_from_directory(
        dataset_path,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,  # No class labels are needed for prediction
        shuffle=False  # Important to keep the order of predictions consistent
    )

    # Make predictions
    vgg_predictions = vgg_model.predict(generator)
    vgg_predicted_labels = np.argmax(vgg_predictions, axis=1)


    rounded_predictions = np.round(vgg_predictions, decimals=8)
    # print(rounded_predictions)


    ensemble_predictions = 0.5 * (vgg_predictions + vit_predictions)
   
    # give the predicted label 
    ensemble_predicted_labels = np.argmax(ensemble_predictions, axis=1)
    # print(ensemble_predicted_labels)

    # Normalize the path separators in filepaths
    filepaths = [os.path.normpath(path) for path in generator.filepaths]

# Extract labels (0 for RoughBark, 1 for StripeCanker)
    labels = [0 if "RoughBark" in path else 1 for path in filepaths]

    # print(labels)
    accuracy = accuracy_score(labels, ensemble_predicted_labels)
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(labels, ensemble_predicted_labels)

    print(conf_matrix)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=hp["class_names"], yticklabels=hp["class_names"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    vit_predicted_labels = np.argmax(vit_predictions, axis=1)

    # Create a classification report
    report = classification_report(labels, ensemble_predicted_labels)
    vit_report = classification_report(labels, vit_predicted_labels)

# Print the classification report

    print("report for vit scratch")
    print(vit_report)
    print("report for ensemble")
    print(report)

# print("Ensemble Model Accuracy:", accuracy)
# vgg_accuracy = accuracy_score(labels, vgg_predicted_labels)
# print("VIT-SCRATCH Model Accuracy:", vgg_accuracy)

    fpr, tpr, thresholds = roc_curve(labels, ensemble_predicted_labels)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.2f})'.format(roc_auc))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()





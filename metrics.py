# this file generates confusion matrix, acc graph, loss curve for train and validation 
import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from train import load_data, tf_dataset
from vit import ViT
from tensorflow.keras.models import load_model

# Load the history from file
with open('training_history_patch_16.pkl', 'rb') as file:
    saved_history = pickle.load(file)

# Plot training & validation accuracy values
plt.plot(saved_history['acc'])
plt.plot(saved_history['val_acc'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(saved_history['loss'])
plt.plot(saved_history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# print("Keys in the saved history:", saved_history.keys())

# # CONFUSIONN MATRIX 
# # Extract true labels and predicted labels from the training history
# train_true_indices = np.argmax(saved_history['true_labels'], axis=1)
# train_predicted_indices = np.argmax(saved_history['predicted_labels'], axis=1)

# # Generate confusion matrix for the training set
# conf_matrix_train = confusion_matrix(train_true_indices, train_predicted_indices)

# # Print the confusion matrix for the training set
# print("Confusion Matrix (Training Set):")
# print(conf_matrix_train)

# # Print classification report for training set
# class_names = hp["class_names"]
# print("\nClassification Report (Training Set):")
# print(classification_report(train_true_indices, train_predicted_indices, target_names=class_names))
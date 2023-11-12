import matplotlib.pyplot as plt
import pickle

# Load the history from file
with open('training_history.pkl', 'rb') as file:
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

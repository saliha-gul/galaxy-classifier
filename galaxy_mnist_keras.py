
import h5py
import numpy as np
import tensorflow as tf #Manages the entire workflow from data preprocessing to model training and evaluation
from tensorflow.keras.utils import to_categorical #Converts class labels to binary matrix for multi-class classification
from tensorflow.keras.models import Sequential #Sequential class is a simple way to stack layers linearly to define a model.
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # the layers of the model 
from tensorflow.keras.optimizers import Adam #Adam is an optimization algorthm 
import matplotlib.pyplot as plt #visualization library for plotting graphs

# download the data 
def load_data(h5_file):
    with h5py.File(h5_file, "r") as f:
        images = np.array(f["images"])
        labels = np.array(f["labels"])
    return images, labels

# download the train and test data
train_images, train_labels = load_data("data/GalaxyMNIST/raw/train_dataset.hdf5")
test_images, test_labels = load_data("data/GalaxyMNIST/raw/test_dataset.hdf5")

# Normalization and transform the tensor format
train_images = train_images / 255.0
test_images = test_images / 255.0

# One-hot encoding for labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Verilerin boyutlarını yazdır
print(f"Train Images Shape: {train_images.shape}")
print(f"Train Labels Shape: {train_labels.shape}")
#output:
#Train Images Shape: (8000, 64, 64, 3)
#Train Labels Shape: (8000, 4)

# Modeli oluştur
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)), # 3x3 kernel size, 32 filters
    MaxPooling2D(pool_size=(2, 2)), # 2x2 pooling window
    Conv2D(64, (3, 3), activation='relu'), # 3x3 kernel size, 64 filters
    MaxPooling2D(pool_size=(2, 2)), # 2x2 pooling window
    Flatten(), # flatten the output
    Dense(128, activation='relu'), # 128 neurons
    Dropout(0.5), # dropout rate
    Dense(4, activation='softmax') # 4 output classes  
])


model.compile(optimizer=Adam(learning_rate=0.001),  #To optimize the parameters of the model (to minimize the loss function)
              loss='categorical_crossentropy', # measures the difference between predicted and actual values.
              metrics=['accuracy']) # Evaluates the accuracy of the model


model.summary() #If there is an error between the input and output sizes in the model, model.summary() helps you spot it.

# Modeli eğit
history = model.fit(
    train_images, train_labels, # training data
    validation_data=(test_images, test_labels), # validation data
    epochs=10, #
    batch_size=32
)


test_loss, test_acc = model.evaluate(test_images, test_labels) #Evaluate the model on the test data
print(f"Test Accuracy: {test_acc:.2f}")

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

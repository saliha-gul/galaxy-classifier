import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import Compose, ToTensor, Normalize
from sklearn.metrics import accuracy_score
import h5py

# Veriyi yüklemek için bir fonksiyon
def load_hdf5_dataset(path):
    with h5py.File(path, "r") as f:
        images = torch.tensor(f['images'][:])  # Görseller
        labels = torch.tensor(f['labels'][:])  # Etiketler
    return images, labels

# Verileri yükleme
train_images, train_labels = load_hdf5_dataset('data/GalaxyMNIST/raw/train_dataset.hdf5')
test_images, test_labels = load_hdf5_dataset('data/GalaxyMNIST/raw/test_dataset.hdf5')

# Normalize etmek ve tensöre dönüştürmek için dönüşümler
transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])

# Verileri normalize et ve TensorDataset oluştur
train_images = train_images.permute(0, 3, 1, 2) / 255.0  # Görüntüleri 0-1 aralığına getir
test_images = test_images.permute(0, 3, 1, 2) / 255.0
train_dataset = TensorDataset(train_images, train_labels)
test_dataset = TensorDataset(test_images, test_labels)

# CNN Modelini Tanımlama
class GalaxyCNN(nn.Module):
    def __init__(self):
        super(GalaxyCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)  # Fully connected layer
        self.fc2 = nn.Linear(128, 4)  # 4 sınıf (smooth_round, smooth_cigar, vb.)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)  # Flatten
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# DataLoader
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model ve cihaz
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GalaxyCNN().to(device)

# Kayıp fonksiyonu ve optimizasyon
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Eğitim
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# Test ve doğrulama
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")
torch.save(model.state_dict(), "galaxy_cnn_model.pt")
print("Model kaydedildi: galaxy_cnn_model.pt")

# **************With KERAS**************
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





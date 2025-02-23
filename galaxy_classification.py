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


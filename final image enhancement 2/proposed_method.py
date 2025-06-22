import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --------------- Dataset Class ---------------
class DecomposedImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def gamma_correction(self, img, gamma_value):
        invGamma = 1.0 / gamma_value
        table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(img, table)

    def log_correction(self, img, c=1.0):
        img_log = c * (np.log(1 + (img / 255.0)))
        img_log = np.uint8(np.clip(img_log * 255, 0, 255))
        return img_log

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to fixed size
        img = cv2.resize(img, (128, 128))

        # Generate gamma corrected images
        gamma_imgs = [self.gamma_correction(img, gamma) for gamma in np.linspace(0.5, 2.0, 5)]
        # Generate log corrected images
        log_imgs = [self.log_correction(img, c) for c in np.linspace(0.5, 2.0, 5)]

        # Stack all
        all_imgs = gamma_imgs + log_imgs
        all_imgs = [img/255.0 for img in all_imgs]  # Normalize [0,1]
        all_imgs = np.stack(all_imgs, axis=0)  # Shape (10, 128, 128, 3)

        # Change to torch tensor and rearrange dims
        all_imgs = torch.tensor(all_imgs, dtype=torch.float32).permute(0, 3, 1, 2)  # (10, 3, 128, 128)

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return all_imgs, label

    def __len__(self):
        return len(self.image_paths)


# --------------- CNN Model ---------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(30, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 32 * 32, 256)
        self.fc2 = nn.Linear(256, 2)  # Binary classification

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# --------------- Load Dataset ---------------
# Dummy example paths
image_folder = '/path/to/your/images'  # TODO: Replace with your path
image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.jpg', '.png'))]

# Dummy labels (0 or 1)
labels = np.random.randint(0, 2, size=len(image_paths))

train_paths, val_paths, train_labels, val_labels = train_test_split(image_paths, labels, test_size=0.2, random_state=42)

train_dataset = DecomposedImageDataset(train_paths, train_labels)
val_dataset = DecomposedImageDataset(val_paths, val_labels)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)


# --------------- Train the Model ---------------
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.view(images.size(0), -1, 128, 128).to(device)  # Merge 10 images
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# --------------- Evaluate the Model ---------------
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in val_loader:
        images = images.view(images.size(0), -1, 128, 128).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Validation Accuracy: {100 * correct / total:.2f}%')

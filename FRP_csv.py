import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# -------- Config --------
CSV_PATH = r"D:\FRP_project\fer2013.csv"   # CHANGE ONLY IF NEEDED
BATCH = 64
EPOCHS = 20
LR = 1e-3
NUM_CLASSES = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ------------------------

# Extra: Check file exists and is readable
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV not found at: {CSV_PATH}")

try:
    open(CSV_PATH, 'r').close()
except:
    raise PermissionError("File cannot be opened. Close Excel/Notepad and remove Read-Only.")

# Dataset class for CSV
class FER2013Dataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pixels = np.fromstring(row['pixels'], dtype=int, sep=' ').astype(np.uint8)
        pixels = pixels.reshape(48, 48)
        img = Image.fromarray(pixels, mode='L')
        if self.transform: img = self.transform(img)
        return img, int(row['emotion'])

# Transforms
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(48, scale=(0.9,1.0)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

transform_val = transforms.Compose([
    transforms.CenterCrop(48),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load CSV
df = pd.read_csv(CSV_PATH)

train_df = df[df['Usage'] == 'Training']
val_df   = df[df['Usage'] == 'PublicTest']

train_ds = FER2013Dataset(train_df, transform=transform_train)
val_ds   = FER2013Dataset(val_df, transform=transform_val)

# Handle imbalance
class_counts = np.bincount(train_df['emotion'])
weights = 1.0 / (class_counts + 1e-6)
sample_weights = [weights[y] for y in train_df['emotion']]
sampler = WeightedRandomSampler(sample_weights,
                                num_samples=len(sample_weights),
                                replacement=True)

train_loader = DataLoader(train_ds, batch_size=BATCH, sampler=sampler)
val_loader   = DataLoader(val_ds, batch_size=BATCH, shuffle=False)

# CNN Model
class SimpleFER(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

model = SimpleFER().to(DEVICE)

criterion = nn.CrossEntropyLoss(torch.tensor(weights, dtype=torch.float).to(DEVICE))
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Training
for epoch in range(EPOCHS):
    model.train()
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            out = model(imgs.to(DEVICE))
            preds.extend(out.argmax(1).cpu().numpy())
            trues.extend(labels.numpy())

    acc = np.mean(np.array(preds) == np.array(trues))
    print(f"Epoch {epoch} — Val Accuracy: {acc:.4f}")

torch.save(model.state_dict(), "fer_model.pth")
print("Model saved.")

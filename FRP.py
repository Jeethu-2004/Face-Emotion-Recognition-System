# fer_training.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# ---------- Config ----------
DATA_DIR = "path_to_fer2013_csv_or_extracted"  # adapt
BATCH = 64
EPOCHS = 30
LR = 1e-3
NUM_CLASSES = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Simple CNN ----------
class SimpleFER(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# ---------- Data: If you have CSV from Kaggle, implement a custom Dataset; else use folder structure ----------
# For brevity, here is an example using ImageFolder with pre-saved 48x48 PNGs for each class.
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

train_ds = datasets.ImageFolder(os.path.join(DATA_DIR,"train"), transform=transform_train)
val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR,"val"), transform=transform_val)

# weighted sampler to handle imbalance
class_counts = np.bincount([y for _,y in train_ds.samples])
weights = 1.0 / class_counts
sample_weights = [weights[y] for _,y in train_ds.samples]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_ds, batch_size=BATCH, sampler=sampler, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=4)

# ---------- Model, loss, optimizer ----------
model = SimpleFER(num_classes=NUM_CLASSES).to(DEVICE)
# compute class weights for loss (alternative to sampler)
class_weights = torch.tensor((1.0 / class_counts), dtype=torch.float).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)

# ---------- Training loop ----------
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for imgs, labels in tqdm(train_loader, desc=f"Train {epoch}"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    train_loss = running_loss / len(train_loader.dataset)

    # validation
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            out = model(imgs)
            pred = out.argmax(dim=1).cpu().numpy()
            preds.extend(pred.tolist())
            trues.extend(labels.numpy().tolist())

    report = classification_report(trues, preds, digits=4, output_dict=True)
    val_acc = report['accuracy']
    print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_acc={val_acc:.4f}")
    scheduler.step(val_acc)

# ---------- Save model ----------
torch.save(model.state_dict(), "simple_fer.pth")

# After training, compute confusion matrix and per-class metrics for failure analysis:
print(classification_report(trues, preds))
print(confusion_matrix(trues, preds))

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models, transforms
from PIL import Image

SEED = 42
DATA_DIR = "data/train"
MODEL_PATH = "models/resnet50_finetuned.pth"
NUM_CLASSES = 100
BATCH_SIZE = 32
EPOCHS = 30
LR_HEAD = 1e-3
LR_BACKBONE = 1e-4
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.1
EARLY_STOP_PATIENCE = 5

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class ImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.samples = []
        self.transform = transform
        class_names = sorted(
            [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))],
            key=lambda x: int(x)
        )
        for label, cls in enumerate(class_names):
            cls_dir = os.path.join(data_dir, cls)
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(cls_dir, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


full_dataset = ImageDataset(DATA_DIR)
n = len(full_dataset)
indices = list(range(n))
random.shuffle(indices)
split = int(n * (1 - VAL_SPLIT))
train_idx, val_idx = indices[:split], indices[split:]

train_set = Subset(ImageDataset(DATA_DIR, train_transform), train_idx)
val_set = Subset(ImageDataset(DATA_DIR, val_transform), val_idx)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"Train: {len(train_set)}, Val: {len(val_set)}")

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(device)

backbone_params = [p for name, p in model.named_parameters() if "fc" not in name]
head_params = list(model.fc.parameters())

optimizer = torch.optim.AdamW([
    {"params": backbone_params, "lr": LR_BACKBONE},
    {"params": head_params, "lr": LR_HEAD},
], weight_decay=WEIGHT_DECAY)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss()

best_val_acc = 0.0
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)

    train_acc = correct / total
    train_loss = total_loss / total

    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total += images.size(0)

    val_acc = val_correct / val_total
    scheduler.step()

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | val_acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"  -> Saved best model (val_acc: {best_val_acc:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

print(f"\nBest val accuracy: {best_val_acc:.4f}")
print(f"Model saved to {MODEL_PATH}")

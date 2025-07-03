import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms, models
from torchvision.models import resnet18, ResNet18_Weights
import csv

# === CONFIGURATION ===
DATA_DIR = Path("data/processed")
BATCH_SIZE = 64
NUM_CLASSES = 43
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
IMG_SIZE = (64, 64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model(num_classes=NUM_CLASSES):
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def get_data_loaders(batch_size=BATCH_SIZE, subset=False, val_split=0.2):
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Capture correct class ordering from base train
    base_train = datasets.ImageFolder(DATA_DIR / 'train')
    classes, class_to_idx = base_train.classes, base_train.class_to_idx

    # Load full training with transforms
    full_train = datasets.ImageFolder(DATA_DIR / 'train', transform=train_transform)
    full_train.classes = classes
    full_train.class_to_idx = class_to_idx

    if subset:
        full_train = Subset(full_train, list(range(3000)))

    val_size = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size
    train_ds, val_ds = random_split(full_train, [train_size, val_size])

    test_ds = datasets.ImageFolder(DATA_DIR / 'test', transform=test_transform)
    test_ds.classes = classes
    test_ds.class_to_idx = class_to_idx
    if subset:
        test_ds = Subset(test_ds, list(range(1000)))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    epoch_loss = running_loss / len(loader.dataset)
    print(f"Train Loss: {epoch_loss:.4f}")
    return epoch_loss

def evaluate(model, loader, criterion, device):
    model.eval()
    correct = total = 0
    running_loss = 0.0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    epoch_loss = running_loss / len(loader.dataset)
    accuracy = correct / total
    print(f"Test Loss: {epoch_loss:.4f}, Test Acc: {accuracy:.4f}")
    return epoch_loss, accuracy

def main():
    # Initialize CSV logging
    log_path = Path("training_log.csv")
    with open(log_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_acc"])

    print(f"Using device: {DEVICE}")
    train_loader, val_loader, test_loader = get_data_loaders(subset=False, val_split=0.2)
    print(f"Train samples:      {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples:       {len(test_loader.dataset)}")

    model = get_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        print("  → Validation:")
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)

        # Log to CSV
        with open(log_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch+1, f"{train_loss:.4f}", f"{val_loss:.4f}", f"{val_acc:.4f}"])

        scheduler.step()

    print("\nFinal Test Evaluation:")
    evaluate(model, test_loader, criterion, DEVICE)

    # Save model
    ckpt_dir = Path("models")
    ckpt_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "traffic_resnet18.pth")
    print("Training complete, model saved to models/traffic_resnet18.pth")

if __name__ == "__main__":
    main()

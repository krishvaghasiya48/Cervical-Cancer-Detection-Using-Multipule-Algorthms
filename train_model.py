import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import ResNet50_Weights, VGG19_Weights
from torch.utils.data import DataLoader, WeightedRandomSampler
import splitfolders
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

from estan import ESTAN
from collections import Counter
import numpy as np

class CYENET(nn.Module):
    def __init__(self, num_classes=2):
        super(CYENET, self).__init__()
        # Simple CNN architecture as placeholder for CYENET
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def main():
    # Parameters
    input_size = 224
    batch_size = 32
    epochs = 40
    num_classes = 2  # Binary classification: cancer vs no cancer
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model selection: 'resnet50', 'vgg19', 'cyenet', 'estan'
    model_name = 'resnet50'  # Change this to select model

    # Data augmentation and transforms
    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
        transforms.RandomResizedCrop(input_size, scale=(0.7, 1.0)),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Dataset locations
    base_dir = r'D:\\projects\\cervical canser detector\\data_binary'  # User should prepare binary labeled data here
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')

    def folder_has_images(folder):
        if not os.path.exists(folder):
            return False
        for class_folder in os.listdir(folder):
            class_path = os.path.join(folder, class_folder)
            if os.path.isdir(class_path):
                for file in os.listdir(class_path):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')):
                        return True
        return False

    def check_original_data(base_dir):
        # Check if original data has class folders with images
        if not os.path.exists(base_dir):
            raise RuntimeError(
                f"Data directory {base_dir} does not exist. "
                "Please prepare your binary labeled data in this directory with class subfolders containing images."
            )
        found = False
        for class_folder in os.listdir(base_dir):
            class_path = os.path.join(base_dir, class_folder)
            if os.path.isdir(class_path):
                for file in os.listdir(class_path):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')):
                        found = True
        return found

    # Step 1: Ensure original data exists and is valid
    if not check_original_data(base_dir):
        raise RuntimeError(
            f"No valid images found in {base_dir}. "
            "Please ensure your data folder contains at least two class subfolders (e.g., 'cancer', 'no_cancer'), each with at least one image file, before running this script."
        )

    # Step 2: Automatically split only if needed
    if not (folder_has_images(train_dir) and folder_has_images(val_dir)):
        # Remove old train/val if they exist (to avoid nested folders)
        for folder in [train_dir, val_dir]:
            if os.path.exists(folder):
                import shutil
                shutil.rmtree(folder)
        print("Splitting data into train/val folders...")
        splitfolders.ratio(
            base_dir,
            output=base_dir,
            seed=42,
            ratio=(.8, .2)
        )
    else:
        print("Train/val folders with images already exist. Skipping split.")

    # Step 3: Check again after split
    if not (folder_has_images(train_dir) and folder_has_images(val_dir)):
        raise RuntimeError(
            "After splitting, no images found in train/val folders. "
            "Check your original data structure and file types."
        )

    # Step 4: Datasets and loaders
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    # Handle class imbalance with WeightedRandomSampler
    train_targets = [label for _, label in train_dataset.imgs]
    class_counts = Counter(train_targets)
    class_weights = {cls: 1.0/count for cls, count in class_counts.items()}
    samples_weights = [class_weights[label] for label in train_targets]
    sampler = WeightedRandomSampler(samples_weights, num_samples=len(samples_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Step 5: Load selected model and modify final layer for binary classification
    if model_name == 'resnet50':
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 1)
    elif model_name == 'vgg19':
        model = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, 1)
    elif model_name == 'cyenet':
        model = CYENET(num_classes=1)
    elif model_name == 'estan':
        model = ESTAN(num_classes=1)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    # Implement focal loss for imbalanced data and hard examples
    class FocalLoss(nn.Module):
        def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
            super(FocalLoss, self).__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.reduction = reduction
            self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

        def forward(self, inputs, targets):
            bce_loss = self.bce_loss(inputs, targets)
            pt = torch.exp(-bce_loss)
            focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
            if self.reduction == 'mean':
                return focal_loss.mean()
            elif self.reduction == 'sum':
                return focal_loss.sum()
            else:
                return focal_loss

    criterion = FocalLoss()

    # Optional: TensorBoard writer for monitoring
    writer = SummaryWriter()

    # Early stopping parameters
    patience = 7
    best_val_loss = float('inf')
    epochs_no_improve = 0

    def calculate_metrics(outputs, labels):
        preds = torch.sigmoid(outputs) >= 0.5
        preds_np = preds.cpu().numpy()
        labels_np = labels.cpu().numpy()
        precision = precision_score(labels_np, preds_np)
        recall = recall_score(labels_np, preds_np)
        f1 = f1_score(labels_np, preds_np)
        return precision, recall, f1

    # Step 6: Training loop with early stopping and checkpointing
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_outputs = []
        all_labels = []
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            preds = torch.sigmoid(outputs) >= 0.5
            correct += (preds == labels.byte()).sum().item()
            total += labels.size(0)
            all_outputs.append(outputs.detach())
            all_labels.append(labels.detach())
        train_loss = running_loss / total
        train_acc = correct / total
        train_precision, train_recall, train_f1 = calculate_metrics(torch.cat(all_outputs), torch.cat(all_labels))

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_outputs = []
        val_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                preds = torch.sigmoid(outputs) >= 0.5
                val_correct += (preds == labels.byte()).sum().item()
                val_total += labels.size(0)
                val_outputs.append(outputs)
                val_labels.append(labels)
        val_loss /= val_total
        val_acc = val_correct / val_total
        val_precision, val_recall, val_f1 = calculate_metrics(torch.cat(val_outputs), torch.cat(val_labels))

        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}, Train F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")

        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch+1)
        writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch+1)
        writer.add_scalars('Precision', {'train': train_precision, 'val': val_precision}, epoch+1)
        writer.add_scalars('Recall', {'train': train_recall, 'val': val_recall}, epoch+1)
        writer.add_scalars('F1', {'train': train_f1, 'val': val_f1}, epoch+1)

        scheduler.step(val_loss)

        # Early stopping and checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), f'cervical_cancer_detection_binary_{model_name}.pth')
            print("Model checkpoint saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    writer.close()
    print(f"Model training complete and saved as 'cervical_cancer_detection_binary_{model_name}.pth'.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error during training: {e}")

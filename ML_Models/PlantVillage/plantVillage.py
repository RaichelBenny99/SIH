import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
import os
import time

# --- 1. CONFIGURATION ---
# IMPORTANT: Change this path to where your PlantVillage dataset is located!
DATA_DIR = "./Plant_leave_diseases_dataset_with_augmentation" 

# Model and training parameters
MODEL_SAVE_PATH = "plant_disease_model.pth"
NUM_EPOCHS = 10  # Start with 10, increase if you have time
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# --- 2. SETUP DEVICE (GPU or CPU) ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device.upper()}")

# --- 3. DATA PREPARATION ---
# Define transformations for the training and validation sets
# Training transform includes data augmentation to make the model more robust
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Validation transform just resizes and normalizes
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("Loading dataset...")
# Load dataset twice so each split gets the correct transform without
# the shared-object bug (random_split Subsets share the same .dataset).
train_full = datasets.ImageFolder(DATA_DIR, transform=train_transform)
val_full   = datasets.ImageFolder(DATA_DIR, transform=val_transform)

# Get class names
class_names = train_full.classes
num_classes = len(class_names)
print(f"Found {num_classes} classes.")

# Split 80/20 with a fixed seed so export_logits.py can reproduce
# the exact same validation split for temperature-scaling calibration.
total = len(train_full)
train_size = int(0.8 * total)
val_size = total - train_size
generator = torch.Generator().manual_seed(42)
all_indices = torch.randperm(total, generator=generator).tolist()
train_dataset = Subset(train_full, all_indices[:train_size])
val_dataset   = Subset(val_full,   all_indices[train_size:])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
print("Data preparation complete.")

# --- 4. MODEL DEFINITION (TRANSFER LEARNING) ---
# Load a pre-trained EfficientNet-B0 model
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

# Freeze all the parameters in the feature extraction part of the model
for param in model.parameters():
    param.requires_grad = False

# Replace the final layer (the classifier) with a new one for our specific number of classes
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, num_classes)

# Move the model to the selected device
model = model.to(device)
print("Model setup complete.")

# --- 5. LOSS FUNCTION AND OPTIMIZER ---
criterion = nn.CrossEntropyLoss()
# Only train the parameters of the new classifier layer
optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)

# --- 6. TRAINING LOOP ---
def train_model():
    print("Starting training...")
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # --- Validation Phase ---
        model.eval()
        running_loss = 0.0
        running_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(val_dataset)
        epoch_acc = running_corrects.double() / len(val_dataset)
        print(f"Validation Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    # --- 7. SAVE THE TRAINED MODEL ---
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    time_elapsed = time.time() - start_time
    print(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Model saved to: {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    train_model()
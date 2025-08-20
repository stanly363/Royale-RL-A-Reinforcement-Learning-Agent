import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import os
import time
import copy
import numpy as np

# --- Configuration ---
DATA_PATH = "sorted_data/cards"
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 25
MODEL_SAVE_PATH = "hand_classifier_best.pth"

def train_model():
    # --- 1. Check for GPU and Set Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"SUCCESS: Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA not found. Using CPU.")

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE), transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(IMG_SIZE), transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    print("Loading dataset...")
    full_dataset = datasets.ImageFolder(DATA_PATH)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    train_dataset.dataset.transform = data_transforms['train']
    val_dataset.dataset.transform = data_transforms['val']
    dataloaders = {
        'train': torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True),
        'val': torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    }
    class_names = full_dataset.classes
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")
    with open("class_names.txt", "w") as f:
        f.write("\n".join(class_names))
    

    print("Building model...")
    model = models.mobilenet_v2(weights='IMAGENET1K_V1')
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # --- 4. The Optimized Training Loop ---
    print("\nStarting optimized training...")
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    epochs_no_improve = 0
    patience = 2

    use_amp = device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)


    try:
        for epoch in range(EPOCHS):
            # (The inner loop code for training and validation is the same)...
            print(f'Epoch {epoch+1}/{EPOCHS}')
            print('-' * 10)
            for phase in ['train', 'val']:
                model.train() if phase == 'train' else model.eval()
                running_loss = 0.0
                running_corrects = 0
                for inputs, labels in dataloaders[phase]:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train':
                scheduler.step()
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                    print(f"    -> New best model weights stored with validation loss: {best_loss:.4f}")
                else:
                    epochs_no_improve += 1
                    print(f"    -> Validation loss did not improve for {epochs_no_improve} epoch(s).")
            
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs.")
                break
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving best model...")

    finally:

        time_elapsed = time.time() - since
        print(f'\nTraining finished in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Loss: {best_loss:4f}')

        # --- 5. Save the Best Model ---
        model.load_state_dict(best_model_wts)
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Best model weights saved to '{MODEL_SAVE_PATH}'")


if __name__ == '__main__':
    train_model()

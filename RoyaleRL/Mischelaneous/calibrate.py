
import torch
from torchvision import models, transforms, datasets
import os
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
# Adjust these paths to match your project structure
MODEL_PATH = "hand_classifier_best.pth"
DATA_DIR = "sorted_data/cards" # Point this to your test/validation folder
CLASS_NAMES_PATH = "card_names.txt" # The file with your card names, one per line
IMG_SIZE = 128
BATCH_SIZE = 16
# ---------------------

def test_model(model, dataloader, device):
    """Calculates the accuracy of the model on the test dataset."""
    model.eval()  # Set the model to evaluation mode
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad(): # No need to calculate gradients during testing
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            correct_predictions += torch.sum(preds == labels.data)
            total_samples += len(inputs)

    accuracy = (correct_predictions.double() / total_samples) * 100
    print(f"\n--- Model Performance ---")
    print(f"Accuracy on test set: {accuracy:.2f}%")
    return accuracy

def visualize_predictions(model, dataloader, class_names, device, num_images=16):
    """Shows a grid of images with their predicted and true labels."""
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(12, 12))

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 4, 4, images_so_far)
                ax.axis('off')
                
                prediction_name = class_names[preds[j]]
                true_name = class_names[labels[j]]

                title = f"Pred: {prediction_name}\nTrue: {true_name}"
                color = "green" if prediction_name == true_name else "red"
                ax.set_title(title, color=color)
                
                # We need to un-normalize the image to display it correctly
                img = inputs.cpu().data[j].numpy().transpose((1, 2, 0))
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean
                img = np.clip(img, 0, 1)
                
                ax.imshow(img)

                if images_so_far == num_images:
                    plt.tight_layout()
                    plt.show()
                    return

if __name__ == '__main__':
    # 1. Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Load class names
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    num_classes = len(class_names)
    print(f"Found {num_classes} classes.")

    # 3. Load the trained model
    print(f"Loading model from {MODEL_PATH}...")
    # Assuming MobileNetV2 architecture as used in your vision script
    model = models.mobilenet_v2()
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)
    print("Model loaded successfully.")

    # 4. Prepare the dataset
    data_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image_dataset = datasets.ImageFolder(DATA_DIR, data_transform)
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Loaded {len(image_dataset)} images from {DATA_DIR}")

    # 5. Run the tests
    test_model(model, dataloader, device)
    visualize_predictions(model, dataloader, class_names, device)
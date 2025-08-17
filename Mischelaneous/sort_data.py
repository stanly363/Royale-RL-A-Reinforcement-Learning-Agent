import os
from PIL import Image
from torchvision import transforms
import random

# --- Configuration ---
# The path to your dataset that needs balancing
DATA_PATH = "sorted_sprites/"

# The target number of images for each card folder
TARGET_COUNT = 100

def augment_and_save(image_path, save_path, transform):
    """Loads an image, applies a transform, and saves the new image."""
    try:
        image = Image.open(image_path).convert("RGB")
        augmented_image = transform(image)
        augmented_image.save(save_path)
    except Exception as e:
        print(f"    - Could not process {os.path.basename(image_path)}: {e}")

def main():
    """
    Balances the dataset by oversampling minority classes using data augmentation.
    """
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data path not found: {DATA_PATH}")
        return

    # --- Define the set of random transformations to apply ---
    augmentation_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    ])
    
    print("Starting dataset balancing...")

    # Iterate through each class folder (e.g., 'knight', 'arrows')
    for class_name in os.listdir(DATA_PATH):
        class_path = os.path.join(DATA_PATH, class_name)
        if not os.path.isdir(class_path):
            continue

        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        current_count = len(image_files)

        print(f"\nProcessing folder: '{class_name}'")
        print(f"Found {current_count} images.")

        # If the folder has fewer images than our target, augment it
        if current_count > 0 and current_count < TARGET_COUNT:
            num_to_generate = TARGET_COUNT - current_count
            print(f"Augmenting with {num_to_generate} new images...")

            for i in range(num_to_generate):
                # Choose a random original image from the folder to augment
                source_image_name = random.choice(image_files)
                source_image_path = os.path.join(class_path, source_image_name)
                
                # Create a new unique filename
                base_name, extension = os.path.splitext(source_image_name)
                new_filename = f"{base_name}_aug_{i}{extension}"
                save_path = os.path.join(class_path, new_filename)
                
                # Apply the augmentation and save the new image
                augment_and_save(source_image_path, save_path, augmentation_transform)
        else:
            print("This folder already has enough images. Skipping.")

    print("\nDataset balancing complete!")

if __name__ == '__main__':
    main()
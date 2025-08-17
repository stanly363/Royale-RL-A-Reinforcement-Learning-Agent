import cv2
import os
import shutil
import numpy as np

# --- Configuration ---
# The parent folder containing the already sorted subfolders
SORTED_DATA_PATH = "sorted_data/"

# The folder containing images that failed the first sorting pass
REJECT_PATH = os.path.join(SORTED_DATA_PATH, "_rejects")

# A new threshold for this refinement pass. You might want it to be slightly
# lower than the first pass, as we're comparing less-than-perfect images.
REFINEMENT_THRESHOLD = 0.8
def preload_sorted_templates(path):
    """
    Loads all already-sorted images into memory.
    This is much faster than reading from disk in every loop.
    """
    all_templates = {}
    print("Pre-loading all sorted images into memory...")
    # List all class folders (e.g., 'knight', 'arrows')
    class_folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f)) and not f.startswith('_')]
    
    for class_name in class_folders:
        all_templates[class_name] = []
        class_folder_path = os.path.join(path, class_name)
        for filename in os.listdir(class_folder_path):
            if filename.endswith('.png'):
                img = cv2.imread(os.path.join(class_folder_path, filename), 0)
                if img is not None:
                    all_templates[class_name].append(img)
    
    print(f"Pre-loaded images for {len(all_templates)} classes.")
    return all_templates

def main():
    """
    Tries to sort rejected images by comparing them against all already-sorted images.
    """
    if not os.path.exists(REJECT_PATH):
        print(f"No reject folder found at '{REJECT_PATH}'. Nothing to do.")
        return

    # Pre-load all sorted images for faster comparison
    sorted_templates = preload_sorted_templates(SORTED_DATA_PATH)
    
    # Get a list of all the images to refine
    rejected_files = [f for f in os.listdir(REJECT_PATH) if f.endswith('.png')]
    total_files = len(rejected_files)
    if total_files == 0:
        print("No rejected images to process. Exiting.")
        return
        
    print(f"\nFound {total_files} images to refine. This may take a while...")
    
    # --- Main Refinement Loop ---
    for i, filename in enumerate(rejected_files):
        img_path = os.path.join(REJECT_PATH, filename)
        image_to_sort = cv2.imread(img_path, 0)
        
        if image_to_sort is None:
            print(f"[{i+1}/{total_files}] Could not read {filename}, skipping.")
            continue

        best_overall_score = -1
        best_overall_class = None

        # Compare the rejected image against every pre-loaded sorted image
        for class_name, template_list in sorted_templates.items():
            for template_img in template_list:
                if template_img.shape[0] > image_to_sort.shape[0] or template_img.shape[1] > image_to_sort.shape[1]:
                    continue
                
                res = cv2.matchTemplate(image_to_sort, template_img, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)
                
                if max_val > best_overall_score:
                    best_overall_score = max_val
                    best_overall_class = class_name
        
        # --- Move the File ---
        if best_overall_score >= REFINEMENT_THRESHOLD:
            # A confident match was found! Move the file.
            destination_folder = os.path.join(SORTED_DATA_PATH, best_overall_class)
            shutil.move(img_path, os.path.join(destination_folder, filename))
            print(f"[{i+1}/{total_files}] Refined {filename} as '{best_overall_class}' (Best Score: {best_overall_score:.2f})")
        else:
            # Still no confident match found.
            print(f"[{i+1}/{total_files}] Could not refine {filename} (Best score: {best_overall_score:.2f}). Leaving in rejects.")

    print("\nRefinement pass complete!")

if __name__ == '__main__':
    main()
import os
import time
from PIL import ImageGrab


from scaler import Scaler
import config

# How often to take a screenshot
CAPTURE_INTERVAL_SECONDS = 1 

def get_user_choices():

    while True:
        try:
            prompt = "Enter two card slot numbers (1-4), separated by a space: "
            choices = input(prompt).split()
            
            if len(choices) != 2:
                print("Error: Please enter exactly two numbers.")
                continue
            
            slot1 = int(choices[0])
            slot2 = int(choices[1])
            
            if not (1 <= slot1 <= 4 and 1 <= slot2 <= 4):
                print("Error: Please enter numbers between 1 and 4.")
                continue

            if slot1 == slot2:
                print("Error: Please enter two different numbers.")
                continue
            
            # Convert user-friendly 1-4 to zero-based index 0-3
            return (slot1 - 1, slot2 - 1)

        except ValueError:
            print("Error: Invalid input. Please enter numbers only.")

def main():

    try:
        # Use your Scaler class to find the exact game area
        scaler = Scaler()
    except ValueError as e:
        print(f"ERROR: Could not initialize the scaler. {e}")
        return
    
    # Get user input for which cards to capture
    slot_index1, slot_index2 = get_user_choices()
    
    # Create output directories
    dir1 = f"new_card_{slot_index1 + 1}"
    dir2 = f"new_card_{slot_index2 + 1}"
    os.makedirs(dir1, exist_ok=True)
    os.makedirs(dir2, exist_ok=True)
    print(f"\nSaving images to '{dir1}' and '{dir2}'...")
    print("Press CTRL+C to stop capturing.")

    # --- Correctly calculate the absolute screen coordinates ---

    # 1. Get the top-left corner of the detected game area
    game_area_left, game_area_top, _, _ = scaler.game_area_rect
    
    # 2. Get the relative offsets for the chosen slots from config
    relative_box1 = config.CARD_OFFSETS_WITH_SIZE[slot_index1]
    relative_box2 = config.CARD_OFFSETS_WITH_SIZE[slot_index2]
    
    # 3. Scale the relative boxes to the current game area size
    scaled_box1 = scaler.scale_box(relative_box1)
    scaled_box2 = scaler.scale_box(relative_box2)

    # 4. Calculate the final absolute bounding box for the screenshot
    #    bbox = game_area_offset + scaled_card_offset
    abs_bbox1 = (
        game_area_left + scaled_box1[0], 
        game_area_top  + scaled_box1[1], 
        game_area_left + scaled_box1[0] + scaled_box1[2], 
        game_area_top  + scaled_box1[1] + scaled_box1[3]
    )
    abs_bbox2 = (
        game_area_left + scaled_box2[0], 
        game_area_top  + scaled_box2[1], 
        game_area_left + scaled_box2[0] + scaled_box2[2], 
        game_area_top  + scaled_box2[1] + scaled_box2[3]
    )
    
    counter1, counter2 = 0, 0
    try:
        while True:
            # --- Capture and save for the first slot ---
            img1 = ImageGrab.grab(bbox=abs_bbox1)
            save_path1 = os.path.join(dir1, f"capture_{counter1:04d}.png")
            img1.save(save_path1)
            counter1 += 1
            
            # --- Capture and save for the second slot ---
            img2 = ImageGrab.grab(bbox=abs_bbox2)
            save_path2 = os.path.join(dir2, f"capture_{counter2:04d}.png")
            img2.save(save_path2)
            counter2 += 1
            
            # Use \r to overwrite the same line in the terminal
            print(f"\rCaptured: {counter1} images for Slot {slot_index1 + 1} | {counter2} images for Slot {slot_index2 + 1}", end="")
            
            time.sleep(CAPTURE_INTERVAL_SECONDS)
            
    except KeyboardInterrupt:
        print("\n\nStopping capture process.")
    finally:
        print(f"Total images saved in '{dir1}': {counter1}")
        print(f"Total images saved in '{dir2}': {counter2}")

if __name__ == "__main__":
    main()

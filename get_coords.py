import cv2
import numpy as np
import pyautogui
from PIL import ImageGrab
import os
import time

def find_elixir_number(screenshot_region, elixir_templates):
    """
    Searches a screenshot region for the best-matching elixir number template.
    """
    screenshot_gray = cv2.cvtColor(np.array(screenshot_region), cv2.COLOR_RGB2GRAY)
    
    best_match = None
    best_val = 0.8
    
    for value, template in elixir_templates.items():
        if template is None:
            continue
            
        res = cv2.matchTemplate(screenshot_gray, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        
        if max_val > best_val:
            best_val = max_val
            best_match = value
            
    return int(best_match) if best_match else None

def main():
    # Define the path to your elixir number templates
    template_dir = r"sorted_data/elixir"
    
    # Define the path to your anchor template
    anchor_path = r"sorted_data/anchors/FourCards.png"

    # Load anchor and elixir templates
    anchor_template = cv2.imread(anchor_path, cv2.IMREAD_GRAYSCALE)
    if anchor_template is None:
        print("Fatal Error: Anchor template not found. Exiting.")
        return

    elixir_templates = {}
    print("Loading elixir number templates...")
    for i in range(11):
        path = os.path.join(template_dir, f"{i}.png")
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            elixir_templates[str(i)] = img
        else:
            print(f"Warning: Elixir template not found at {path}")
    
    if not elixir_templates:
        print("Fatal Error: No elixir templates were loaded. Exiting.")
        return

    # Use the offsets you provided.
    x_offset, y_offset, width, height = 11, 144, 386, 38
    
    print("Elixir locator starting. Press Ctrl+C to exit.")
    
    try:
        while True:
            # Take a screenshot
            screenshot_pil = ImageGrab.grab()
            screenshot_gray = cv2.cvtColor(np.array(screenshot_pil), cv2.COLOR_RGB2GRAY)
            
            # Find the anchor
            res_anchor = cv2.matchTemplate(screenshot_gray, anchor_template, cv2.TM_CCOEFF_NORMED)
            _, max_val_anchor, _, anchor_loc = cv2.minMaxLoc(res_anchor)

            if max_val_anchor > 0.8:
                # Calculate the absolute coordinates of the elixir search box
                elixir_left = anchor_loc[0] + x_offset
                elixir_top = anchor_loc[1] + y_offset
                elixir_right = elixir_left + width
                elixir_bottom = elixir_top + height
                
                elixir_box_coords = (elixir_left, elixir_top, elixir_right, elixir_bottom)
                
                # Crop the screenshot to the specific elixir area.
                elixir_region = screenshot_pil.crop(elixir_box_coords)
                
                # Find the elixir number within the cropped region.
                elixir_value = find_elixir_number(elixir_region, elixir_templates)
                
                if elixir_value is not None:
                    print(f"Detected Elixir: {elixir_value}")
                else:
                    print("Elixir number not detected.")
            else:
                print("Anchor not found on screen.")
                
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\nScript stopped by user.")

if __name__ == "__main__":
    main()
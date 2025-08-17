import pyautogui
import time
import cv2
import numpy as np
from PIL import ImageGrab

class Controller:
    """Handles all mouse interactions with the game."""
    def __init__(self):
        # The controller no longer needs to know card slots at initialization.
        print("Controller initialized.")

    def click(self, x, y):
        """Moves to and clicks a given coordinate."""
        pyautogui.moveTo(x, y, duration=0.1)
        pyautogui.click()
        time.sleep(0.5)

    def play_card(self, card_click_x, card_click_y, placement_coords):
        """
        Plays a card by clicking its center, then clicking the placement location.
        """
        print(f"CONTROLLER: Clicking card at ({card_click_x}, {card_click_y})")
        self.click(card_click_x, card_click_y)
        
        print(f"CONTROLLER: Placing card at {placement_coords}")
        self.click(placement_coords[0], placement_coords[1])

    def find_and_click(self, template_img, confidence=0.85):

        # Grab a screenshot and convert it for OpenCV processing
        screen = ImageGrab.grab()
        screen_cv = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2GRAY)

        # Perform template matching
        res = cv2.matchTemplate(screen_cv, template_img, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val >= confidence:
            # Get the width and height of the template
            h, w = template_img.shape
            # Calculate the center of the found region
            center_x = max_loc[0] + w // 2
            center_y = max_loc[1] + h // 2
            
            print(f"Found template with confidence {max_val:.2f}. Clicking center at ({center_x}, {center_y})")
            self.click(center_x, center_y)
            return True
        else:
            print(f"Could not find template with sufficient confidence (Max val: {max_val:.2f})")
            return False
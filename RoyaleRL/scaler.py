# filename: scaler.py
import cv2
import numpy as np
import pyautogui
import pygetwindow as gw
from PIL import ImageGrab
import config
import os # Make sure this is imported

class Scaler:
    def __init__(self):
        # Find the BlueStacks window by a flexible title search
        try:
            bluestacks_windows = gw.getWindowsWithTitle('BlueStacks App Player')
            if not bluestacks_windows:
                raise ValueError("No BlueStacks window found with a title starting with 'BlueStacks App Player'.")
            
            bluestacks_window = bluestacks_windows[0]
            bluestacks_window.activate() 
        except IndexError:
            raise ValueError("BlueStacks window not found. Please ensure the emulator is running and the title is correct.")

        # Find the game area within the BlueStacks window 
        game_area = self._find_game_area_by_combined_methods(bluestacks_window)
        
        if not game_area:
            raise ValueError("Could not find the game area within the BlueStacks window.")

        # The current resolution is the dimensions of the detected game area
        self.game_area_rect = game_area
        self.current_resolution = (self.game_area_rect[2], self.game_area_rect[3])
        
        # Calculate scaling factors based on the game area dimensions
        self.x_scale = self.current_resolution[0] / config.REFERENCE_RESOLUTION[0]
        self.y_scale = self.current_resolution[1] / config.REFERENCE_RESOLUTION[1]
        
        print(f"Scaler initialized. Detected Clash Royale resolution: {self.current_resolution}")
        print(f"Scaling factors: X={self.x_scale:.2f}, Y={self.y_scale:.2f}")

    def _find_game_area_by_combined_methods(self, bluestacks_window):

        bluestacks_rect = (bluestacks_window.left, bluestacks_window.top, bluestacks_window.width, bluestacks_window.height)
        screenshot_pil = ImageGrab.grab(bbox=bluestacks_rect)
        screenshot_np = np.array(screenshot_pil)
        
        # --- Find the Top Boundary using Color Scan ---
        screenshot_hsv = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2HSV)
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(screenshot_hsv, lower_blue, upper_blue)
        center_x = screenshot_np.shape[1] // 2
        top_edge_y = 0
        try:
            for y in range(screenshot_np.shape[0]):
                if blue_mask[y, center_x] == 0:
                    top_edge_y = y
                    break
        except IndexError: pass
        if top_edge_y == 0:
            print("Could not find the top edge via color scan. Assuming fullscreen/borderless.")
            top_edge_y = 0

        # --- Find Horizontal Boundaries using Contours ---
        # Crop the screenshot to ignore the top timer area for contour detection
        cropped_screenshot = screenshot_np[top_edge_y + 100:, :]
        cropped_screenshot_gray = cv2.cvtColor(cropped_screenshot, cv2.COLOR_RGB2GRAY)
        ret, thresh = cv2.threshold(cropped_screenshot_gray, 10, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
        if len(contours) < 2: return None
        
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        left_box = min(bounding_boxes, key=lambda b: b[0])
        right_box = max(bounding_boxes, key=lambda b: b[0])
        
        x_start = left_box[0] + left_box[2]
        width = right_box[0] - x_start
        if width <= 0: return None
            
        # --- Calculate Final Bounding Box ---
        # The height is calculated from the original top_edge_y
        final_height = bluestacks_window.height - top_edge_y

        return (bluestacks_window.left + x_start, bluestacks_window.top + top_edge_y, width, final_height)

    def scale_coords(self, coords):
        """Scales coordinates relative to the found game area."""
        return (int((coords[0] - self.game_area_rect[0]) * self.x_scale), 
                int((coords[1] - self.game_area_rect[1]) * self.y_scale))
    
    def scale_box(self, box):
        """Scales a box tuple (x, y, w, h)."""
        x, y, w, h = box
        scaled_x = int(x * self.x_scale)
        scaled_y = int(y * self.y_scale)
        scaled_w = int(w * self.x_scale)
        scaled_h = int(h * self.y_scale)
        return (scaled_x, scaled_y, scaled_w, scaled_h)

    def scale_template(self, template_path):
        """Loads and scales a template image."""
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            raise FileNotFoundError(f"Template not found at {template_path}")
        
        new_w = int(template.shape[1] * self.x_scale)
        new_h = int(template.shape[0] * self.y_scale)
        
        scaled_template = cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return scaled_template

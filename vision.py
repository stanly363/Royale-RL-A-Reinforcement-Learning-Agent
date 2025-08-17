# filename: vision.py
import torch
from torchvision import models, transforms
from PIL import Image
import config
import time
import cv2
import os
import numpy as np
import easyocr

# --- Component 1: The Logical Elixir Tracker ---
class ElixirTracker:
    """Tracks elixir logically and syncs with visual reads."""
    def __init__(self):
        self.current_elixir = 0.0
        self.tracking_started = False
        self.last_update_time = None

    def start(self):
        if not self.tracking_started:
            self.tracking_started = True
            self.current_elixir = 7.0
            self.last_update_time = time.time()
            print("--- Hand detected! Starting elixir tracking at 7. ---")

    def reset(self):
        if self.tracking_started:
            self.tracking_started = False
            self.current_elixir = 0.0
            self.last_update_time = None
            print("--- Elixir tracking paused. ---")

    def update(self, elapsed_time):
        if not self.tracking_started: return
        self.current_elixir += elapsed_time * (1.0 / 2.8)
        if self.current_elixir > 10: self.current_elixir = 10.0
        
    def sync_with_vision(self, visual_elixir):
        """Corrects the logical count only if the whole number is wrong."""
        if self.tracking_started and visual_elixir is not None and isinstance(visual_elixir, (int, float)):
            logical_elixir_int = int(self.current_elixir)
            visual_elixir_int = int(visual_elixir)
            if logical_elixir_int != visual_elixir_int:
                print(f"SYNC: Correcting elixir from {self.current_elixir:.1f} to {float(visual_elixir):.1f}")
                self.current_elixir = float(visual_elixir)

# --- Component 2: The AI Card Classifier ---
class CardClassifier:
    """Handles loading the ML model and making predictions."""
    def __init__(self, model_path, class_names_path, device):
        self.device = device
        with open(class_names_path, "r") as f: self.class_names = [line.strip() for line in f.readlines()]
        self.transform = self._get_transform()
        self.model = self._load_model(model_path, len(self.class_names))

    def _load_model(self, path, num_classes):
        model = models.mobilenet_v2()
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
        model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        model.to(self.device)
        model.eval()
        print("Model loaded successfully.")
        return model

    def _get_transform(self):
        return transforms.Compose([
            transforms.Resize(config.IMG_SIZE), transforms.CenterCrop(config.IMG_SIZE),
            transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict_batch(self, images):
        tensors = [self.transform(img) for img in images]
        if not tensors: return []
        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            outputs = self.model(batch)
            _, preds = torch.max(outputs, 1)
        return [self.class_names[p] for p in preds]
    
# --- New ElixirVision Class (Integrated) ---
class ElixirVision:
    """
    Handles elixir detection using template matching.
    """
    def __init__(self):
        self.template_dir = r"C:\Users\stanl\Downloads\Royale-RL-A-Reinforcement-Learning-Agent\sorted_data\elixir"
        self.templates = self._load_elixir_templates()
        if not self.templates:
            raise FileNotFoundError("Elixir templates not found. Bot cannot run.")

    def _load_elixir_templates(self):
        templates = {}
        for i in range(11):
            path = os.path.join(self.template_dir, f"{i}.png")
            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                templates[str(i)] = img
            else:
                print(f"Warning: Elixir template not found at {path}")
        return templates

    def get_elixir_value(self, screenshot_region):
        screenshot_gray = cv2.cvtColor(np.array(screenshot_region), cv2.COLOR_RGB2GRAY)
        
        best_match = None
        best_val = 0.8
        
        for value, template in self.templates.items():
            if template is None:
                continue

            res = cv2.matchTemplate(screenshot_gray, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            
            if max_val > best_val:
                best_val = max_val
                best_match = value
                
        return int(best_match) if best_match else None

# --- Main Vision Class: This ties everything together ---
class Vision:
    """A single class to handle all vision-related tasks."""
    def __init__(self):
        print("Initializing Vision...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier = CardClassifier(config.MODEL_PATH, config.CLASS_NAMES_PATH, self.device)
        self.elixir_tracker = ElixirTracker()
        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        self.elixir_vision = ElixirVision()
        self.last_visual_check_time = time.time()
        self.last_known_hand = []
        self.last_update_time = time.time()
    
    def _draw_debug_overlay(self, screenshot_cv, anchor_pos, card_coordinates):
        """Draws all OCR and card boxes on a screenshot for debugging."""
        anchor_x, anchor_y = anchor_pos

        # Draw the anchor for reference
        cv2.rectangle(screenshot_cv, 
                      (anchor_x, anchor_y),
                      (anchor_x + 449, anchor_y + 25), # Hardcoded anchor size
                      (0, 255, 0), 2)
        cv2.putText(screenshot_cv, "ANCHOR", (anchor_x, anchor_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw OCR regions (RED)
        for box in config.OCR_OFFSETS:
            x = anchor_x + box['x_offset']
            y = anchor_y + box['y_offset']
            w = box['width']
            h = box['height']
            cv2.rectangle(screenshot_cv, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(screenshot_cv, box['name'], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Draw card slots (BLUE)
        for i, box in enumerate(card_coordinates):
            x, y, w, h = box
            cv2.rectangle(screenshot_cv, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(screenshot_cv, f"CARD {i+1}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display the final image
        window_name = "Live Debugger"
        cv2.imshow(window_name, screenshot_cv)
        cv2.waitKey(1)

    def _get_ocr_values(self, screenshot, anchor_pos):
        """Performs OCR on all predefined regions."""
        anchor_x, anchor_y = anchor_pos
        ocr_results = {}
        
        for box in config.OCR_OFFSETS:
            x = anchor_x + box['x_offset']
            y = anchor_y + box['y_offset']
            width = box['width']
            height = box['height']

            region_image = screenshot.crop((x, y, x + width, y + height))
            img_np = np.array(region_image)

            # Use different methods for Elixir vs. other OCR
            if box['name'] == 'Elixir':
                elixir_val = self.elixir_vision.get_elixir_value(region_image)
                ocr_results[box['name']] = elixir_val
            else:
                result = self.reader.readtext(img_np, allowlist='0123456789:')
                text = ' '.join([res[1] for res in result]).strip()
                ocr_results[box['name']] = text
            
        return ocr_results

    def get_game_state(self, screenshot, anchor_pos, card_coordinates):
        """
        The main method to get the current hand, elixir, and OCR data.
        """
        current_time = time.time()
        elapsed_time = current_time - self.last_update_time
        self.elixir_tracker.update(elapsed_time) 
        self.last_update_time = current_time

        # OCR for tower health, time, etc.
        ocr_data = self._get_ocr_values(screenshot, anchor_pos)
        
        # Calculate elixir region using anchor and offset
        anchor_x, anchor_y = anchor_pos
        elixir_x = anchor_x + config.ELIXIR_OFFSET['x']
        elixir_y = anchor_y + config.ELIXIR_OFFSET['y']
        elixir_w = config.ELIXIR_OFFSET['width']
        elixir_h = config.ELIXIR_OFFSET['height']
        
        elixir_region = screenshot.crop((elixir_x, elixir_y, elixir_x + elixir_w, elixir_y + elixir_h))
        
        # Get the visual elixir value
        visual_elixir = self.elixir_vision.get_elixir_value(elixir_region)
        
        if visual_elixir is not None:
            self.elixir_tracker.sync_with_vision(visual_elixir)

        cards_to_predict = []
        hand_area_bbox = self._calculate_hand_bbox(card_coordinates)
        if hand_area_bbox:
            hand_area_pil = screenshot.crop(hand_area_bbox)
            for (x, y, w, h) in card_coordinates:
                relative_x = x - hand_area_bbox[0]
                relative_y = y - hand_area_bbox[1]
                bbox = (relative_x, relative_y, relative_x + w, relative_y + h)
                cards_to_predict.append(hand_area_pil.crop(bbox))

        current_hand = self.classifier.predict_batch(cards_to_predict)
        
        invalid_slots = current_hand.count('empty') + current_hand.count('Unknown')
        if invalid_slots >= 2: 
            self.elixir_tracker.reset()
        elif invalid_slots == 0 and not self.elixir_tracker.tracking_started: 
            self.elixir_tracker.start()

        if current_hand != self.last_known_hand:
            self.last_known_hand = current_hand
            
        return {
                "hand": self.last_known_hand,
                "elixir": self.elixir_tracker.current_elixir,
                "ocr_data": ocr_data
        }
    
    def _calculate_hand_bbox(self, slots):
        if not slots: return None
        x_coords, y_coords, widths, heights = zip(*slots)
        x_min, y_min = min(x_coords), min(y_coords)
        x_max = max(x + w for x, w in zip(x_coords, widths))
        y_max = max(y + h for y, h in zip(y_coords, heights))
        padding = 15
        return (x_min - padding, y_min - padding, x_max + padding, y_max + padding)

# --- This block allows you to test this script by itself ---
if __name__ == '__main__':
    print("Initializing Vision module for testing...")
    vision_system = Vision()
    print("Press Ctrl+C to stop.")
    try:
        while True:
            # Grab a new screenshot for each loop
            screenshot = ImageGrab.grab()
            # Find the anchor to get coordinates
            try:
                anchor_location = pyautogui.locateOnScreen(config.ANCHOR_IMAGE_PATH, confidence=0.8)
                if anchor_location:
                    anchor_pos = (anchor_location.left, anchor_location.top)
                    # Use a dummy set of card coordinates for testing
                    card_coords = [(anchor_pos[0] + x, anchor_pos[1] + y, w, h) for x, y, w, h in config.CARD_OFFSETS_WITH_SIZE]
                    
                    game_state = vision_system.get_game_state(screenshot, anchor_pos, card_coords)
                    elixir_status = f"{game_state['elixir']:.1f}" if game_state['elixir'] > 0 else "Paused"
                    print(f"Current State -> Hand: {game_state['hand']} | Elixir: {elixir_status} | OCR: {game_state['ocr_data']}")
                    
                    # Draw a live debug overlay
                    screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
                    vision_system._draw_debug_overlay(screenshot_cv, anchor_pos, card_coords)

            except pyautogui.PyAutoGUIException:
                print("Anchor not found, skipping this loop...")

            time.sleep(1)
    except KeyboardInterrupt:
        print("\nVision test finished.")
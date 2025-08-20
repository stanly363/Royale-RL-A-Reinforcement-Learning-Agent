import cv2
import numpy as np
from PIL import ImageGrab
import os
import time
from controller import Controller
import config

class GameStateManager:
    """Manages the game state using visual anchors."""
    def __init__(self, controller, scaler):
        self.controller = controller
        self.scaler = scaler
        self.anchors = self._load_anchors()
        if self.anchors:
            print(f"Loaded {len(self.anchors)} state anchors.")

    def _load_anchors(self):
        anchor_path = "sorted_data/anchors/"
        anchor_files = {
            "MAIN_MENU": "battle_anchor.png",
            "IN_BATTLE": "game_anchor.png",
            "POST_BATTLE": "ok_anchor.png",
            "POST_BATTLE_2": "ok_anchor2.png",
            "WIN_CROWN": "bluecrowns.png",
            "LOSE_CROWN": "redcrowns.png",
            "LUCKY_BOX": "luckybox2.png"
        }

        anchors = {}
        for state, filename in anchor_files.items():
            path = os.path.join(anchor_path, filename)
            if os.path.exists(path):
                anchors[state] = self.scaler.scale_template(path)
        return anchors

    def get_state(self):
        # Grab screenshot of only the game area for efficiency
        game_area_rect = self.scaler.game_area_rect
        
        # Correctly format the bounding box for ImageGrab.grab()
        bbox = (game_area_rect[0], game_area_rect[1], game_area_rect[0] + game_area_rect[2], game_area_rect[1] + game_area_rect[3])
        
        screen_pil = ImageGrab.grab(bbox=bbox)
        screen_cv_gray = cv2.cvtColor(np.array(screen_pil), cv2.COLOR_RGB2GRAY)

        for state, anchor_img in self.anchors.items():
            if anchor_img is None: continue
            res = cv2.matchTemplate(screen_cv_gray, anchor_img, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            if max_val > 0.85:
                return state
        return "UNKNOWN"
    
    def start_match(self):
        print("Attempting to start match...")
        return self.controller.find_and_click('sorted_data/anchors/battle_anchor.png')

    def end_match(self):
        print("Attempting to end match...")
        return self.controller.find_and_click('sorted_data/anchors/ok_anchor.png')
    
    def fix_bug(self):
        print("Attempting to fix...")
        for _ in range(5):
            self.controller.find_and_click('sorted_data/anchors/luckybox2.png')
        return self.controller.find_and_click('sorted_data/anchors/ok_anchor2.png')
    
    def non_max_suppression(self, boxes, scores, overlapThresh):
        if len(boxes) == 0:
            return []
        
        pick = []
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(scores)
        
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            
            overlap = (w * h) / area[idxs[:last]]
            
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
            
        return pick

    def analyze_result(self, screen_pil):
        print("Analyzing match result...")
        screen_cv_gray = cv2.cvtColor(np.array(screen_pil), cv2.COLOR_RGB2GRAY)
        
        win_crown_template = self.anchors.get("WIN_CROWN")
        lose_crown_template = self.anchors.get("LOSE_CROWN")

        if win_crown_template is None or lose_crown_template is None:
            print("Error: Crown anchor templates not found.")
            return
        
        win_w, win_h = win_crown_template.shape[::-1]
        lose_w, lose_h = lose_crown_template.shape[::-1]
        
        threshold = 0.9

        # Define the Region of Interest for your crowns at the bottom 60%
        game_height = screen_pil.height
        my_roi_y = int(game_height * 0.40) 
        my_roi = screen_cv_gray[my_roi_y:game_height, :]

        # Define the Region of Interest for your opponent's crowns at the top 40%
        opponent_roi_y = 0
        opponent_roi_height = int(game_height * 0.40) 
        opponent_roi = screen_cv_gray[opponent_roi_y : opponent_roi_y + opponent_roi_height, :]
        
        # Count my crowns in my ROI
        res_win_me = cv2.matchTemplate(my_roi, win_crown_template, cv2.TM_CCOEFF_NORMED)
        win_locs_me = np.where(res_win_me >= threshold)
        my_win_boxes = self.non_max_suppression(np.array([
            [pt[0], pt[1], pt[0] + win_w, pt[1] + win_h] for pt in zip(*win_locs_me[::-1])
        ]), res_win_me[win_locs_me], 0.3)
        
        # Count my opponent's crowns in my opponent's ROI
        res_win_opponent = cv2.matchTemplate(opponent_roi, win_crown_template, cv2.TM_CCOEFF_NORMED)
        win_locs_opponent = np.where(res_win_opponent >= threshold)
        opponent_win_boxes = self.non_max_suppression(np.array([
            [pt[0], pt[1], pt[0] + win_w, pt[1] + win_h] for pt in zip(*win_locs_opponent[::-1])
        ]), res_win_opponent[win_locs_opponent], 0.3)

        # Count empty crowns on opponent's side
        res_lose_opponent = cv2.matchTemplate(opponent_roi, lose_crown_template, cv2.TM_CCOEFF_NORMED)
        lose_locs_opponent = np.where(res_lose_opponent >= threshold)
        opponent_lose_boxes = self.non_max_suppression(np.array([
            [pt[0], pt[1], pt[0] + lose_w, pt[1] + lose_h] for pt in zip(*lose_locs_opponent[::-1])
        ]), res_lose_opponent[lose_locs_opponent], 0.3)
        
        # The logic is now based on win crowns only
        my_crowns = len(my_win_boxes)
        opponent_crowns = len(opponent_win_boxes)

        print(f"Detected my crowns: {my_crowns}")
        print(f"Detected opponent's crowns: {opponent_crowns}")
        
        if my_crowns > opponent_crowns:
            print(f"You won! 🎉 ({my_crowns}-{opponent_crowns})")
        elif opponent_crowns > my_crowns:
            print(f"You lost. 😔 ({my_crowns}-{opponent_crowns})")
        else:
            print(f"The match was a draw. 🤝 ({my_crowns}-{opponent_crowns})")
    
    def get_crown_boxes(self, screen_pil):
        screen_cv_gray = cv2.cvtColor(np.array(screen_pil), cv2.COLOR_RGB2GRAY)
        
        # Load the templates for crowns.
        win_crown_template = self.anchors.get("WIN_CROWN")
        lose_crown_template = self.anchors.get("LOSE_CROWN")
        
        if win_crown_template is None or lose_crown_template is None:
            return [], [] # Return empty lists if templates are missing

        win_w, win_h = win_crown_template.shape[::-1]
        lose_w, lose_h = lose_crown_template.shape[::-1]
        
        threshold = 0.9

        # Define the Region of Interest for your crowns at the bottom 60%
        game_height = screen_pil.height
        my_roi_y = int(game_height * 0.40) 
        my_roi = screen_cv_gray[my_roi_y:game_height, :]

        # Define the Region of Interest for your opponent's crowns at the top 40%
        opponent_roi_y = 0
        opponent_roi_height = int(game_height * 0.40) 
        opponent_roi = screen_cv_gray[opponent_roi_y : opponent_roi_y + opponent_roi_height, :]

        # --- Count my crowns in my ROI ---
        res_win_me = cv2.matchTemplate(my_roi, win_crown_template, cv2.TM_CCOEFF_NORMED)
        win_locs_me = np.where(res_win_me >= threshold)
        my_win_boxes = self.non_max_suppression(np.array([
            [pt[0], pt[1], pt[0] + win_w, pt[1] + win_h] for pt in zip(*win_locs_me[::-1])
        ]), res_win_me[win_locs_me], 0.3)
        
        # --- Count opponent's crowns in opponent's ROI ---
        res_win_opponent = cv2.matchTemplate(opponent_roi, win_crown_template, cv2.TM_CCOEFF_NORMED)
        win_locs_opponent = np.where(res_win_opponent >= threshold)
        opponent_win_boxes = self.non_max_suppression(np.array([
            [pt[0], pt[1], pt[0] + win_w, pt[1] + win_h] for pt in zip(*win_locs_opponent[::-1])
        ]), res_win_opponent[win_locs_opponent], 0.3)

        # We also need to check for lose crowns, but only on the opponent's side
        res_lose_opponent = cv2.matchTemplate(opponent_roi, lose_crown_template, cv2.TM_CCOEFF_NORMED)
        lose_locs_opponent = np.where(res_lose_opponent >= threshold)
        opponent_lose_boxes = self.non_max_suppression(np.array([
            [pt[0], pt[1], pt[0] + lose_w, pt[1] + lose_h] for pt in zip(*lose_locs_opponent[::-1])
        ]), res_lose_opponent[lose_locs_opponent], 0.3)

        # The analyze_result function will use these counts
        my_crowns = len(my_win_boxes)
        opponent_crowns = len(opponent_win_boxes)

        # Return the boxes to be used for the debug overlay
        return my_win_boxes, opponent_win_boxes
    
import cv2
import numpy as np
from PIL import ImageGrab
import os
import time
# config is no longer needed here for button coords
from controller import Controller

class GameStateManager:
    """Manages the game state using visual anchors."""
    def __init__(self, controller):
        self.controller = controller # Use the controller that is passed in
        self.anchors = self._load_anchors()
        if self.anchors:
            print(f"Loaded {len(self.anchors)} state anchors.")


    def _load_anchors(self):
        anchor_path = "sorted_data/anchors/"
        anchor_files = {
            "MAIN_MENU": "battle_anchor.png",
            "IN_BATTLE": "game_anchor.png",
            "POST_BATTLE": "ok_anchor.png",
            "WIN_CROWN": "win_anchor.png",
        "LOSE_CROWN": "lose_anchor.png"  
    }

        anchors = {}
        for state, filename in anchor_files.items():
            path = os.path.join(anchor_path, filename)
            if os.path.exists(path):
                anchors[state] = cv2.imread(path, 0)
        return anchors

    def get_state(self):
        screen_pil = ImageGrab.grab()
        screen_cv_gray = cv2.cvtColor(np.array(screen_pil), cv2.COLOR_RGB2GRAY)
        for state, anchor_img in self.anchors.items():
            res = cv2.matchTemplate(screen_cv_gray, anchor_img, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            if max_val > 0.85:
                return state
        return "UNKNOWN"
    
    def start_match(self):
        """Starts a new match by finding and clicking the battle button."""
        print("Attempting to start match...")
        # Use the anchor for the main menu as the template to find
        return self.controller.find_and_click(self.anchors.get("MAIN_MENU"))

    def end_match(self):
        """Ends the match by finding and clicking the OK button."""
        print("Attempting to end match...")
        # Use the anchor for the post-battle screen as the template to find
        return self.controller.find_and_click(self.anchors.get("POST_BATTLE"))
    
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

    def analyze_result(self):
        # ... (Your existing analyze_result method)
        print("Analyzing match result...")
        screen_pil = ImageGrab.grab()
        screen_cv_gray = cv2.cvtColor(np.array(screen_pil), cv2.COLOR_RGB2GRAY)
        
        win_crown_template = self.anchors.get("WIN_CROWN")
        lose_crown_template = self.anchors.get("LOSE_CROWN")

        if win_crown_template is None or lose_crown_template is None:
            print("Error: Crown anchor templates not found.")
            return
        
        win_w, win_h = win_crown_template.shape[::-1]
        lose_w, lose_h = lose_crown_template.shape[::-1]
        
        threshold = 0.9  
        
        res_win = cv2.matchTemplate(screen_cv_gray, win_crown_template, cv2.TM_CCOEFF_NORMED)
        win_locs = np.where(res_win >= threshold)
        
        win_boxes = []
        win_scores = []
        for pt in zip(*win_locs[::-1]):
            win_boxes.append([pt[0], pt[1], pt[0] + win_w, pt[1] + win_h])
            win_scores.append(res_win[pt[1], pt[0]])
        win_boxes = np.array(win_boxes)
        win_scores = np.array(win_scores)
        
        # Now the call is correct
        win_pick = self.non_max_suppression(win_boxes, win_scores, 0.3)
        win_count = len(win_pick)

        res_lose = cv2.matchTemplate(screen_cv_gray, lose_crown_template, cv2.TM_CCOEFF_NORMED)
        lose_locs = np.where(res_lose >= threshold)
        
        lose_boxes = []
        lose_scores = []
        for pt in zip(*lose_locs[::-1]):
            lose_boxes.append([pt[0], pt[1], pt[0] + lose_w, pt[1] + lose_h])
            lose_scores.append(res_lose[pt[1], pt[0]])
        lose_boxes = np.array(lose_boxes)
        lose_scores = np.array(lose_scores)

        # Now the call is correct
        lose_pick = self.non_max_suppression(lose_boxes, lose_scores, 0.3)
        lose_count = len(lose_pick)
        
        print(f"Detected {win_count} win crowns and {lose_count} lose crowns.")
        
        if win_count > lose_count:
            print("You won! 🎉")
        elif lose_count > win_count:
            print("You lost. 😔")
        else:
            print("Draw or no clear winner detected.")
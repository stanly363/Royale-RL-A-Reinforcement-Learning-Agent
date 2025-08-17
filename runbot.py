# filename: run_bot.py
import time
import cv2
import pyautogui
import numpy as np
from PIL import ImageGrab

from game_state_manager import GameStateManager
from controller import Controller
from vision import Vision
import config

def find_cards_dynamically(template_img):
    """
    Takes one screenshot to find the anchor and return
    the screenshot, anchor position, and card coordinates.
    """
    screenshot_pil = ImageGrab.grab()
    screenshot_cv = cv2.cvtColor(np.array(screenshot_pil), cv2.COLOR_RGB2GRAY)
    
    result = cv2.matchTemplate(screenshot_cv, template_img, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val >= 0.8:
        anchor_x, anchor_y = max_loc
        card_boxes = []
        for offset_x, offset_y, size_w, size_h in config.CARD_OFFSETS_WITH_SIZE:
            card_x = anchor_x + offset_x
            card_y = anchor_y + offset_y
            card_boxes.append((card_x, card_y, size_w, size_h))
        return screenshot_pil, (anchor_x, anchor_y), card_boxes
    else:
        return None, None, []

def wait_for_state_change(state_manager, initial_state, timeout=20):
    """Waits until the game state is different from the initial state."""
    print(f"Waiting for state to change from '{initial_state}'...")
    start_time = time.time()
    while True:
        current_state = state_manager.get_state()
        if current_state != initial_state:
            print(f"State changed to '{current_state}'.")
            return
        if time.time() - start_time > timeout:
            print("Wait timed out.")
            return
        time.sleep(0.5)

def decide_action(game_state):
    """
    The 'Brain' of the bot. This is where you will add your strategies.
    It now has access to OCR data.
    """
    hand = game_state['hand']
    elixir = game_state['elixir']
    ocr_data = game_state['ocr_data']
    
    print(f"BRAIN: Received state - Hand: {hand}, Elixir: {elixir:.1f}, OCR Data: {ocr_data}")

    if 'giant' in hand and elixir >= 5:
        card_slot_index = hand.index('giant')
        placement_coords = (980, 500) 
        print("BRAIN: Decision: Play Giant")
        return {'action': 'play_card', 'card_slot': card_slot_index, 'position': placement_coords}
    
    return None

def main():
    # --- Setup ---
    controller = Controller()
    state_manager = GameStateManager(controller=controller) 
    vision = Vision()
    
    # Store the card coordinates and anchor position for the duration of the battle
    battle_coords = {"cards": None, "anchor_pos": None, "screenshot": None, "debug_shown": False}

    try:
        anchor_template = cv2.imread('sorted_data/anchors/FourCards.png', 0)
        if anchor_template is None:
            raise FileNotFoundError("Could not find 'FourCards.png'. Bot cannot run.")
    except FileNotFoundError as e:
        print(f"FATAL ERROR: {e}")
        return

    print("Bot starting... Press Ctrl+C in the terminal to stop.")
    
    try:
        while True:
            current_state = state_manager.get_state()
            print(f"\nCurrent game state: {current_state}")
            
            if current_state == "IN_BATTLE":
                if battle_coords["cards"] is None:
                    print("First time in battle state, locating cards...")
                    # Wait 3 seconds to ensure the game is fully loaded
                    print("Waiting for 3 seconds to ensure the game screen is stable...")
                    time.sleep(3)
                    
                    screenshot, anchor_pos, card_coords = find_cards_dynamically(anchor_template)
                    if card_coords:
                        battle_coords["screenshot"] = screenshot
                        battle_coords["anchor_pos"] = anchor_pos
                        battle_coords["cards"] = card_coords
                        print(f"Found {len(battle_coords['cards'])} card locations.")

                        # --- ONE-SHOT DEBUGGER ---
                        if config.DEBUG_VISUALS and not battle_coords["debug_shown"]:
                            print("Drawing a single debug overlay. Press a key to continue...")
                            screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
                            vision._draw_debug_overlay(screenshot_cv, battle_coords["anchor_pos"], battle_coords["cards"])
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()
                            battle_coords["debug_shown"] = True # Set flag to prevent showing again

                if battle_coords["cards"]:
                    # Get a fresh screenshot at the start of each loop
                    screenshot = ImageGrab.grab()
                    
                    current_game_state = vision.get_game_state(
                        screenshot, 
                        battle_coords["anchor_pos"], 
                        battle_coords["cards"]
                    )
                    action = decide_action(current_game_state)
                    
                    if action and action['action'] == 'play_card':
                        slot = action['card_slot']
                        pos = action['position']
                        box = battle_coords["cards"][slot]
                        click_x, click_y = box[0] + box[2] // 2, box[1] + box[3] // 2
                        controller.play_card(click_x, click_y, pos)
                else:
                    print("Could not find cards on screen (anchor not visible).")
            
            elif current_state == "POST_BATTLE":
                print("Match has ended. Returning to main menu...")
                battle_coords = {"cards": None, "anchor_pos": None, "debug_shown": False}
                state_manager.analyze_result()
                state_manager.end_match()
                wait_for_state_change(state_manager, "POST_BATTLE")

            elif current_state == "MAIN_MENU":
                if battle_coords["cards"] is not None:
                    battle_coords = {"cards": None, "anchor_pos": None, "debug_shown": False}
                print("On main menu. Attempting to start a match...")
                state_manager.start_match()
                wait_for_state_change(state_manager, "MAIN_MENU")
            
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nBot stopped by user.")

if __name__ == '__main__':
    main()
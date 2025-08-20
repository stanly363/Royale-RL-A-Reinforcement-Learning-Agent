# filename: main.py
import time
import cv2
import json
import os
import torch
from scaler import Scaler
from game_state_manager import GameStateManager
from controller import Controller
from vision import Vision
from agent import Agent
import config
from config import get_health_percentage, CARD_TO_INDEX, NUM_CARD_TYPES, CARD_COSTS
import pickle
import random
from PIL import ImageGrab

ALL_CARDS = list(CARD_COSTS.keys())
CARD_TO_INDEX = {name: i for i, name in enumerate(ALL_CARDS)}
NUM_CARD_TYPES = len(ALL_CARDS)

# --- UPDATED: Calculate ACTION_DIM based on the grid size in agent.py ---
x_steps = 18
y_steps = 30
NUM_GRID_LOCATIONS = x_steps * y_steps
ACTION_DIM = (NUM_CARD_TYPES + 1) * NUM_GRID_LOCATIONS

STATE_DIM = 1 + 6 + (4 * NUM_CARD_TYPES) + (20 * 4)

def find_cards_dynamically(scaler):
    game_area = scaler.game_area_rect
    bbox = (game_area[0], game_area[1], game_area[0] + game_area[2], game_area[1] + game_area[3])
    screenshot_pil = ImageGrab.grab(bbox=bbox)
    card_boxes = [box for box in config.CARD_OFFSETS_WITH_SIZE]
    return screenshot_pil, (0, 0), card_boxes

def wait_for_state_change(state_manager, initial_state, timeout=30):
    """Waits until the game state is different from the initial state."""
    print(f"Waiting for state to change from '{initial_state}'...")
    start_time = time.time()
    while True:
        current_state = state_manager.get_state()
        if current_state != initial_state:
            print(f"State changed to '{current_state}'.")
            return current_state
        if time.time() - start_time > timeout:
            print("Wait timed out.")
            return "TIMEOUT"
        time.sleep(0.5)

def get_total_health_pct(ocr_data):
    enemy_hp = sum([get_health_percentage(ocr_data.get(k), t) for k, t in [('ptl', 'princess'), ('ptr', 'princess'), ('tk', 'king')]])
    friendly_hp = sum([get_health_percentage(ocr_data.get(k), t) for k, t in [('pbl', 'princess'), ('pbr', 'princess'), ('bk', 'king')]])
    return enemy_hp, friendly_hp

def calculate_reward(last_state, current_state):
    """
    Calculates the reward based on tower damage, enemies destroyed, 
    and a penalty for early King Tower activation.
    """
    ENEMY_DESTROYED_REWARD = 0.1
    KING_TOWER_ACTIVATION_PENALTY = 0.5

    last_ocr = last_state.get('ocr_data', {})
    current_ocr = current_state.get('ocr_data', {})

    last_enemy_hp, last_friendly_hp = get_total_health_pct(last_ocr)
    current_enemy_hp, current_friendly_hp = get_total_health_pct(current_ocr)
    damage_dealt = max(0, last_enemy_hp - current_enemy_hp)
    damage_taken = max(0, last_friendly_hp - current_friendly_hp)
    reward = damage_dealt - damage_taken

    num_enemies_last = len(last_state.get('enemies', []))
    num_enemies_current = len(current_state.get('enemies', []))
    enemies_destroyed = num_enemies_last - num_enemies_current
    
    if enemies_destroyed != 0:
        enemy_reward = enemies_destroyed * ENEMY_DESTROYED_REWARD
        print(f"REWARD: Change in enemy count: {enemies_destroyed}. Applying {enemy_reward:.2f} reward.")
        reward += enemy_reward

    king_was_inactive = not last_ocr.get('tk')
    king_is_now_active = bool(current_ocr.get('tk'))
    ptl_was_alive = bool(last_ocr.get('ptl'))
    ptr_was_alive = bool(last_ocr.get('ptr'))
    
    if king_was_inactive and king_is_now_active and ptl_was_alive and ptr_was_alive:
        print(f"PENALTY: King Tower activated prematurely! Applying -{KING_TOWER_ACTIVATION_PENALTY} penalty.")
        reward -= KING_TOWER_ACTIVATION_PENALTY
        
    return reward

def main():
    scaler = Scaler()
    controller = Controller(scaler=scaler)
    state_manager = GameStateManager(controller=controller, scaler=scaler)
    vision = Vision(scaler=scaler)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # --- UPDATED: Pass the new ACTION_DIM to the Agent ---
    ai_agent = Agent(state_dim=STATE_DIM, action_dim=ACTION_DIM, card_costs=CARD_COSTS, device=device)
    
    print("Bot starting with live learning agent. Press Ctrl+C to stop.")
    
    try:
        while True:
            current_state_name = state_manager.get_state()
            print(f"\nCurrent game state is: {current_state_name}")

            if current_state_name == "MAIN_MENU":
                state_manager.start_match()
                wait_for_state_change(state_manager, "MAIN_MENU")
            
            elif current_state_name == "POST_BATTLE":
                state_manager.end_match()
                wait_for_state_change(state_manager, "POST_BATTLE")

            elif current_state_name == "POST_BATTLE_2":
                state_manager.fix_bug()
                wait_for_state_change(state_manager, "POST_BATTLE_2")

            elif current_state_name == "UNKNOWN":
                time.sleep(5)
            
            elif current_state_name == "IN_BATTLE":
                print("Battle in progress. Agent is playing and learning...")
                current_game_log = {'steps': []}
                last_state, action = None, None
                battle_coords = {"cards": None}
                
                while True:
                    if state_manager.get_state() != "IN_BATTLE":
                        print("Battle has ended.")
                        screenshot, _, _ = find_cards_dynamically(scaler)
                        my_crowns, op_crowns = state_manager.get_crown_boxes(screenshot)
                        final_reward = 1 if len(my_crowns) > len(op_crowns) else -1 if len(op_crowns) > len(my_crowns) else 0
                        
                        if current_game_log['steps']:
                            current_game_log['steps'][-1]['reward'] += final_reward
                            
                            ai_agent.learn_from_game(current_game_log)
                            ai_agent.train(num_epochs=5, batch_size=64)
                            
                        break

                    if battle_coords["cards"] is None:
                        time.sleep(3)
                        _, _, card_coords = find_cards_dynamically(scaler)
                        if card_coords: battle_coords["cards"] = card_coords
                    
                    if battle_coords["cards"]:
                        screenshot, _, _ = find_cards_dynamically(scaler)
                        current_game_state = vision.get_game_state(screenshot, battle_coords["cards"])
                        
                        if last_state and action:
                            reward = calculate_reward(last_state, current_game_state)
                            current_game_log['steps'].append({
                                'state': last_state, 'action': action,
                                'reward': reward, 'next_state': current_game_state
                            })

                        action = ai_agent.decide_action(current_game_state, scaler)
                        if not action:
                            pass
                        
                        elif action.get('action') == 'play_card':
                            slot, pos = action['card_slot'], action['position']
                            
                            if 0 <= slot < len(battle_coords["cards"]):
                                box = battle_coords["cards"][slot]
                                click_x, click_y = box[0] + box[2] // 2, box[1] + box[3] // 2
                                controller.play_card((click_x, click_y), pos)
                            else:
                                print(f"ERROR: AI model predicted an invalid card slot index: {slot}. Random action instead.")
                                action = ai_agent._get_random_action(current_game_state, scaler)
                                if action:
                                    controller.play_card((click_x, click_y), pos)
                        last_state = current_game_state
                    
                    time.sleep(2)
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nBot stopped by user. Saving model and buffer...")
        ai_agent.save()
        ai_agent.save_buffer() # <-- Added this line to save the buffer
        print("Bot stopped.")
        

if __name__ == '__main__':
    main()
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import random
import config
from config import get_health_percentage, CARD_TO_INDEX, NUM_CARD_TYPES, ALL_CARDS, ARENA_BBOX
import pickle
# Define the 18x30 placement grid
x_steps = 18
y_steps = 30
min_x_pct, min_y_pct, max_x_pct, max_y_pct = ARENA_BBOX

PLACEMENT_GRID = []
for i in range(x_steps):
    for j in range(y_steps):
        x_pct = min_x_pct + (max_x_pct - min_x_pct) * (i + 0.5) / x_steps
        y_pct = min_y_pct + (max_y_pct - min_y_pct) * (j + 0.5) / y_steps
        PLACEMENT_GRID.append((x_pct, y_pct))

NUM_GRID_LOCATIONS = len(PLACEMENT_GRID)
ACTION_DIM = (NUM_CARD_TYPES + 1) * NUM_GRID_LOCATIONS # +1 for "do nothing"

class Block(nn.Module):
    def __init__(self, h_dim, n_heads, drop_p):
        super().__init__()
        self.attn = nn.MultiheadAttention(h_dim, n_heads)
        self.ff = nn.Sequential(nn.Linear(h_dim, 4 * h_dim), nn.GELU(), nn.Linear(4 * h_dim, h_dim), nn.Dropout(drop_p))
        self.ln1, self.ln2 = nn.LayerNorm(h_dim), nn.LayerNorm(h_dim)

    def forward(self, x):
        x = x + self.attn(x, x, x)[0]
        x = self.ln1(x)
        x = x + self.ff(x)
        return self.ln2(x)

class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, act_dim, n_blocks, h_dim, context_len, n_heads, drop_p):
        super().__init__()
        self.state_dim, self.act_dim, self.h_dim = state_dim, act_dim, h_dim
        self.embed_state = nn.Linear(self.state_dim, h_dim)
        self.embed_action = nn.Linear(self.act_dim, h_dim)
        self.embed_reward = nn.Linear(1, h_dim)
        self.embed_timestep = nn.Embedding(context_len, h_dim) 
        self.embed_ln = nn.LayerNorm(h_dim)
        self.transformer_blocks = nn.ModuleList([Block(h_dim, n_heads, drop_p) for _ in range(n_blocks)])
        self.predict_action = nn.Sequential(nn.Linear(h_dim, self.act_dim), nn.Tanh())

    def forward(self, states, actions, rewards, timesteps):
        batch_size, seq_len, _ = states.shape
        timesteps = torch.arange(seq_len, device=states.device).unsqueeze(0).repeat(batch_size, 1)
        time_embs = self.embed_timestep(timesteps)
        state_embs = self.embed_state(states) + time_embs
        action_embs = self.embed_action(actions) + time_embs
        reward_embs = self.embed_reward(rewards) + time_embs
        stacked_inputs = torch.stack((state_embs, action_embs, reward_embs), dim=1).permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_len, self.h_dim)
        stacked_inputs = self.embed_ln(stacked_inputs)
        x = stacked_inputs
        for block in self.transformer_blocks:
            x = block(x)
        x = x.reshape(batch_size, seq_len, 3, self.h_dim).permute(0, 2, 1, 3)
        return self.predict_action(x[:,1])

class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = int(capacity)
        self.ptr, self.size = 0, 0
        self.states = np.zeros((self.capacity, state_dim))
        # Store actions as a single integer (dtype int32)
        self.actions = np.zeros(self.capacity, dtype=np.int32)
        self.rewards = np.zeros((self.capacity, 1))
        self.next_states = np.zeros((self.capacity, state_dim))
    
    def add(self, state, action, reward, next_state):
        self.states[self.ptr] = state
        # Now action is just the index
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, context_len):
        if self.size < context_len:
            return None, None, None, None
        indices = np.random.randint(0, self.size - context_len, size=batch_size)
        states, actions, rewards, timesteps = [], [], [], []
        for i in indices:
            states.append(self.states[i:i+context_len])
            # Actions are now a 1-D array of integers
            actions.append(self.actions[i:i+context_len])
            rewards.append(self.rewards[i:i+context_len])
            timesteps.append(np.arange(context_len)) 
        return (torch.tensor(np.array(states), dtype=torch.float32),
                # Convert the integer actions to a tensor
                torch.tensor(np.array(actions), dtype=torch.long),
                torch.tensor(np.array(rewards), dtype=torch.float32),
                torch.tensor(np.array(timesteps), dtype=torch.long))

class Agent:
    def __init__(self, state_dim, action_dim, card_costs, device):
        self.state_dim = state_dim
        self.action_dim = action_dim 
        self.card_costs = card_costs
        self.device = device
        self.model_path = 'rl_agent.pt'
        
        self.epsilon = 1
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99
        
        self.context_len = 10 
        self.n_blocks, self.embed_dim, self.n_heads, self.dropout_p, self.lr = 3, 128, 1, 0.1, 1e-4

        self.model = DecisionTransformer(
            state_dim=state_dim, act_dim=action_dim, n_blocks=self.n_blocks, h_dim=self.embed_dim,
            context_len=self.context_len, n_heads=self.n_heads, drop_p=self.dropout_p
        ).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        
        self.replay_buffer = ReplayBuffer(1e5, self.state_dim, self.action_dim)
        self.load()
        self.buffer_path = 'replay_buffer.pkl' 
        self.replay_buffer = ReplayBuffer(1e5, self.state_dim, self.action_dim)
        
        if os.path.exists(self.buffer_path):
            self.load_buffer()
    
    def save_buffer(self):
        with open(self.buffer_path, 'wb') as f:
            pickle.dump(self.replay_buffer, f)
        print("Replay buffer saved.")

    def load_buffer(self):
        try:
            with open(self.buffer_path, 'rb') as f:
                self.replay_buffer = pickle.load(f)
            print(f"Replay buffer loaded with {self.replay_buffer.size} items.")
        except Exception as e:
            print(f"Could not load replay buffer: {e}")
            print("Starting with a new, empty buffer.")

    def _flatten_state(self, state_dict):
        elixir = np.array([state_dict.get('elixir', 0) / 10.0])
        ocr_data = state_dict.get('ocr_data', {})
        tower_health = np.array([
            get_health_percentage(ocr_data.get('ptl'), 'princess'), get_health_percentage(ocr_data.get('ptr'), 'princess'),
            get_health_percentage(ocr_data.get('tk'), 'king'), get_health_percentage(ocr_data.get('pbl'), 'princess'),
            get_health_percentage(ocr_data.get('pbr'), 'princess'), get_health_percentage(ocr_data.get('bk'), 'king')
        ])
        hand_vector = np.zeros(4 * NUM_CARD_TYPES)
        for i, card in enumerate(state_dict.get('hand', [])):
            if card in CARD_TO_INDEX:
                hand_vector[i * NUM_CARD_TYPES + CARD_TO_INDEX[card]] = 1.0
        enemies_vector = np.zeros(20 * 4)
        enemies = state_dict.get('enemies', [])[:20]
        for i, enemy in enumerate(enemies):
            box = enemy.get('box', (0,0,0,0))
            enemies_vector[i*4:(i+1)*4] = [
                box[0]/config.REFERENCE_RESOLUTION[0], box[1]/config.REFERENCE_RESOLUTION[1],
                box[2]/config.REFERENCE_RESOLUTION[0], box[3]/config.REFERENCE_RESOLUTION[1]
            ]
        return np.concatenate([elixir, tower_health, hand_vector, enemies_vector])

    def _flatten_action(self, action_dict):
        card_slot = action_dict.get('card_slot', None)
        position = action_dict.get('position', None)
        
        # Now returns a single integer, not a vector
        action_index = NUM_CARD_TYPES * NUM_GRID_LOCATIONS 
        if card_slot is not None and position is not None:
            closest_grid_index = self._find_closest_grid_index(position)
            action_index = card_slot * NUM_GRID_LOCATIONS + closest_grid_index
        return action_index

    def _find_closest_grid_index(self, position):
        window_width, window_height = config.REFERENCE_RESOLUTION
        pos_pct = np.array([position[0] / window_width, position[1] / window_height])
        
        placement_grid_np = np.array(PLACEMENT_GRID)
        distances = np.linalg.norm(placement_grid_np - pos_pct, axis=1)
        return np.argmin(distances)

    def decide_action(self, game_state, scaler):
        if np.random.rand() <= self.epsilon:
            print("BRAIN: Choosing a random action (exploring)...")
            return self._get_random_action(game_state, scaler)
        
        print("BRAIN: Using AI model to decide action (exploiting)...")
        return self._get_model_action(game_state, scaler)

    def _get_random_action(self, game_state, scaler):
        hand = game_state.get('hand', [])
        elixir = game_state.get('elixir', 0)
        playable_cards = [i for i, card in enumerate(hand) if card in self.card_costs and elixir >= self.card_costs[card]]
        if not playable_cards:
            return None
        
        card_slot_index = random.choice(playable_cards)
        placement_pct = random.choice(PLACEMENT_GRID)
        
        window_width, window_height = scaler.current_resolution
        placement_x = int(placement_pct[0] * window_width)
        placement_y = int(placement_pct[1] * window_height)
        
        return {'action': 'play_card', 'card_slot': card_slot_index, 'position': (placement_x, placement_y)}

    def _get_model_action(self, game_state, scaler):
        state_vec = self._flatten_state(game_state)
        state_tensor = torch.tensor(state_vec, dtype=torch.float32).reshape(1, 1, self.state_dim).to(self.device)
        
        action_tensor = torch.zeros((1, 1, self.action_dim), dtype=torch.float32).to(self.device)
        reward_tensor = torch.zeros((1, 1, 1), dtype=torch.float32).to(self.device)
        timestep_tensor = torch.tensor([[self.replay_buffer.size % (self.context_len * 3)]], dtype=torch.long).to(self.device)

        self.model.eval()
        with torch.no_grad():
            action_preds = self.model(state_tensor, action_tensor, reward_tensor, timestep_tensor).squeeze(0).squeeze(0)

        probabilities = F.softmax(action_preds, dim=0)
        action_index = torch.argmax(probabilities).item()

        if action_index >= NUM_CARD_TYPES * NUM_GRID_LOCATIONS:
            print("Model chose 'do nothing'.")
            return None

        card_slot_index = action_index // NUM_GRID_LOCATIONS
        grid_location_index = action_index % NUM_GRID_LOCATIONS
        placement_pct = PLACEMENT_GRID[grid_location_index]
        
        window_width, window_height = scaler.current_resolution
        placement_x = int(placement_pct[0] * window_width)
        placement_y = int(placement_pct[1] * window_height)

        return {'action': 'play_card', 'card_slot': card_slot_index, 'position': (placement_x, placement_y)}

    def learn_from_game(self, game_log):
        if not game_log['steps']:
            print("LEARNING: No steps in game log, skipping training.")
            return
        
        print(f"LEARNING: Adding {len(game_log['steps'])} steps to replay buffer...")
        for step in game_log['steps']:
            state_vec = self._flatten_state(step['state'])
            action_vec = self._flatten_action(step['action'])
            reward_val = step['reward']
            next_state_vec = self._flatten_state(step['next_state'])
            self.replay_buffer.add(state_vec, action_vec, reward_val, next_state_vec)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        print(f"LEARNING: Buffer size: {self.replay_buffer.size}. New epsilon: {self.epsilon:.3f}")
        self.save()

    def train(self, num_epochs, batch_size):
        if self.replay_buffer.size < 100: 
            print("TRAINING: Not enough data in replay buffer to start training. Play more games.")
            return
        
        print("Starting agent training...")
        self.model.train()
        start_time = time.time()
        for epoch in range(num_epochs):
            epoch_loss = 0
            for _ in range(100):
                states, actions, rewards, timesteps = self.replay_buffer.sample(batch_size, self.context_len)
                if states is None: continue
                
                states, actions, rewards, timesteps = states.to(self.device), actions.to(self.device), rewards.to(self.device), timesteps.to(self.device)
                
                one_hot_actions = F.one_hot(actions, num_classes=self.action_dim).float()
                
                action_preds = self.model(states, one_hot_actions, rewards, timesteps)

                
                # Use F.cross_entropy and actions as long tensors
                loss = F.cross_entropy(action_preds.view(-1, self.action_dim), actions.view(-1))

                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / 100
            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")
            
            if (epoch + 1) % 10 == 0:
                self.save(f"rl_agent_epoch_{epoch+1}.pt")
                
        print(f"Training complete in {(time.time() - start_time) / 60:.0f}m")
        self.save("rl_agent_final.pt")

    def save(self, path=None):
        save_path = path if path is not None else self.model_path
        torch.save(self.model.state_dict(), save_path)
        print(f"Agent model saved to {save_path}")

    def load(self):
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            print(f"Agent model loaded from {self.model_path}")

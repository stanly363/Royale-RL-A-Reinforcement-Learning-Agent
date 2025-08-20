# Clash Royale AI Bot

This project is a sophisticated AI bot that learns to play the game Clash Royale through reinforcement learning. It uses real-time computer vision to understand the game state and a Decision Transformer model to make strategic decisions.

## Features

* **Advanced AI**: Utilizes a **Decision Transformer**, a modern reinforcement learning architecture, to make decisions based on states, actions, and expected rewards.
* **Real-Time Vision**: Employs a suite of computer vision models to understand the game in real-time:
  * **YOLOv9** for detecting and identifying enemy units.
  * **MobileNetV2** for classifying the cards in the player's hand.
  * **EasyOCR** and template matching for reading tower health, elixir, and game state.
* **Autonomous Learning**: The bot learns from every game it plays. It stores its experiences in a replay buffer and uses this data to continuously improve its strategy.
* **Full Game Control**: Manages the entire game loop, from starting matches and playing cards to handling post-game screens.

## How It Works (Architecture)

The bot operates through a modular system where each component has a specific responsibility. The main loop is orchestrated by `runbot.py`.

1. `runbot.py` (The **Conductor)**
   * Initializes all components.
   * Contains the main `while True` loop that runs the bot.
   * Calls the `GameStateManager` to determine the current screen (e.g., Main Menu, In Battle).
2. `GameStateManager` (The **Navigator)**
   * Takes screenshots of the screen.
   * Uses image templates (anchors) to identify the current game state.
3. **`Vision` (The Eyes)**
   * If the bot is in a battle, the Vision module is activated.
   * It uses its sub-components (`EnemyDetector`, `CardClassifier`, `EasyOCR`) to analyze the screen and build a complete, structured understanding of the game state (tower health, your cards, elixir count, enemy positions).
4. **`Agent` (The Brain)**
   * Receives the game state from the Vision module.
   * Feeds this state into its **Decision Transformer** model to decide the best action (which card to play and where).
   * After each game, it adds the entire match history to its `ReplayBuffer` and retrains its model.
5. **`Controller` (The Hands)**
   * Receives the chosen action from the Agent.
   * Simulates mouse clicks to select the card and deploy it at the specified location on the screen.

## AI Model Architecture Deep Dive

The "brain" of this bot is a **Decision Transformer**. Unlike traditional reinforcement learning methods that learn a policy function (what action to take now), a Decision Transformer is a sequence modeling architecture that predicts future actions based on a desired outcome (i.e., a high reward).

### How The Models Were Built

* **Vision Models**: The initial vision models (`hand_classifier_best.pt`, `best.pt`) were trained on a specific dataset containing only the following 11 cards/units: `archers`, `arrows`, `empty`, `fireball`, `giant`, `goblin_cage`, `goblin_hut`, `knight`, `mini-pekka`, `minions`, and `musketeer`. This means the bot will only recognize these cards out of the box.
* **Reinforcement Learning Agent (`rl_agent.pt`)**: This model was **not pre-trained**. It starts with no knowledge of Clash Royale strategy and learns entirely from scratch through the games it plays ("online learning"). Its initial actions will be random, but they will become more strategic over time as it populates its replay buffer.

### How It Works

1. **Input**: The model takes a sequence of the last few game events as input. Each event consists of a **State** (all the info from the Vision module), the **Action** taken in that state, and the **Reward** received after that action.
2. **Embedding**: These three inputs are converted into numerical vectors (embeddings) and combined with positional information. This tells the model not just *what* happened, but *when* it happened.
3. **Transformer Blocks**: The sequence is then processed by several transformer blocks. These use a mechanism called "self-attention" to weigh the importance of different past events. For example, it might learn that an enemy P.E.K.K.A. played 10 seconds ago is more important than a Skeleton played 2 seconds ago.
4. **Output**: Based on this analysis, the model predicts the next action that is most likely to lead to a high future reward. This action (a card and a location) is then sent to the `Controller`.

This architecture allows the bot to learn complex, long-term strategies by understanding the sequence of events, rather than just reacting to the immediate state of the board.

## Setup and Installation

Follow these steps to get the bot running on your system.

### Prerequisites

* **Python 3.10+**
* An **NVIDIA GPU** with CUDA installed (required for the AI models).
* The game Clash Royale running in an Android emulator (developed and tested with **BlueStacks**).
* **IMPORTANT**: The BlueStacks window must be **fullscreened or windowed fullscreened** to ensure the `scaler.py` script can detect the game area properly.

### Installation Steps

1. **Clone the Repository**
   ```
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```
2. **Create a Python Virtual Environment**
   ```
   python -m venv venv
   ```
3. **Activate the Virtual Environment**
   * On Windows (PowerShell):
     ```
     .\venv\Scripts\Activate.ps1
     ```
   * On macOS/Linux:
     ```
     source venv/bin/activate
     ```
4. **Install Dependencies**
   * The provided `requirements.txt` file contains all necessary packages, including the correct GPU-enabled version of PyTorch.
   ```
   pip install -r requirements.txt
   ```

## How to Run the Bot

Once the setup is complete, ensure the Clash Royale emulator is running in fullscreen and is visible on your screen.

Then, simply run the main script from your activated virtual environment:
```
python runbot.py
```
The bot will find the game window, and you will see status messages printed in the console as it begins to play.

## Extending the Bot

### How to Add a New Card

To teach the bot to recognize and use a new card, you must follow this process carefully. Note that `config.py` already contains the names and elixir costs for all cards in the game.

1. **Collect Images**:
   * Run the `capture_cards.py` script. It will ask you to input two card slot numbers (1-4).
   * Enter the slot number(s) where the new card is currently in your hand.
   * **Crucially**, you need to play other cards to cycle through your deck. This allows the script to capture images of the new card when you have different amounts of elixir available, which slightly changes the card's appearance.
   * Let the script run for several games to collect hundreds of varied images. Press `Ctrl+C` to stop.
2. **Organize the Dataset**:
   * The script will have created new folders (e.g., `new_card_1`).
   * Create a new folder inside `sorted_data/cards/`. The folder name must be the **exact** name of the new card as it appears in `config.py` (e.g., `sorted_data/cards/golem`).
   * Move all the screenshots you collected into this new folder.
3. **Retrain the Card Classifier**:
   * Run the training script to update the `CardClassifier` model:
     ```
     python optimized_train.py
     ```
   * This script automatically generates a `class_names.txt` file.
4. **Verify `class_names.txt`**:
   * **This is a critical step.** Open the newly generated `class_names.txt` file.
   * Confirm that your new card's name is present and that the entire list is in **alphabetical order**.
   * If it's not in order, the bot will misidentify cards. Manually edit the file to fix the order if necessary. The card names here must also match the names in `config.py` exactly.

The bot will now be able to recognize and use the new card.

## How to Improve the Bot

The bot's performance is directly tied to the quality of its models and its reward function. Here are the key areas for improvement:

### 1. Improve the Enemy Detection Model (`best.pt`)

The YOLOv9 model's accuracy is critical. Better enemy detection leads to better decisions.
* **Collect More Data**: The single most effective improvement. Record gameplay in different arenas, against different card levels, and during special events to create a more robust dataset.
* **Refine Annotations**: Ensure your bounding box annotations are tight and accurate.
* **Experiment with Architectures**: Try newer or larger YOLO models (like YOLOv10) which may offer better performance.
* **Tune Hyperparameters**: Adjust learning rate, augmentation settings, and other training parameters to optimize the model's accuracy.

### 2. Enhance the Reward Function

The `calculate_reward` function in `runbot.py` is the heart of the learning process. A more nuanced reward signal can teach the bot more sophisticated strategies.
* **Reward Elixir Advantage**: Give a small positive reward for having more elixir than the opponent and a penalty for having less.
* **Punish Inefficient Plays**: Add a penalty for using a high-cost card on a low-value troop (e.g., using a Rocket on a single Skeleton).
* **Contextual Tower Damage**: Give more reward for tower damage dealt by a high-value troop (like a P.E.K.K.A.) than by a low-cost troop.

### 3. Tune the Decision Transformer

The agent's "brain" can also be tuned.
* **Adjust Context Length**: Experiment with a longer `context_len` in `agent.py` to allow the model to see a longer history of moves when making a decision.
* **Modify Model Size**: Change the `embed_dim`, `n_heads`, or `n_blocks` to make the model larger (potentially smarter, but slower) or smaller (faster, but potentially less effective).

## Recommended Usage

* **Consistency is Key**: Run the bot on the same emulator with the same resolution each time to ensure the vision system works reliably.
* **Initial Learning**: Let the bot play at least 50-100 games on its own. This will populate the `replay_buffer.pkl` file with a diverse set of experiences for it to learn from.
* **Dedicated Training**: While the bot learns a little after each game, its learning is shallow. For deep, meaningful improvement, periodically stop the bot and run a dedicated training session using the data it has collected.

## Troubleshooting

* **Error: "BlueStacks window not found"**: Make sure the emulator is running and that the window title in `scaler.py` exactly matches the title of your emulator window.
* **Error: `pygetwindow.PyGetWindowException`**: This is a permissions error. **Run** your script as an **Administrator**.
* **Error: `Activate.ps1 cannot be loaded...`**: This is a PowerShell security policy. Run this command in PowerShell once to fix it: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process`.

## Disclaimer

This project is for educational and research purposes only. Automating gameplay may be against the terms of service of some games. Use this software responsibly and at your own risk.

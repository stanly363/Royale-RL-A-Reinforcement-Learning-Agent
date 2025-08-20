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

***

## How It Works (Architecture)

The bot operates through a modular system where each component has a specific responsibility. The main loop is orchestrated by `runbot.py`.

1.  **`runbot.py` (The Conductor)**
    * Initializes all components.
    * Contains the main `while True` loop that runs the bot.
    * Calls the `GameStateManager` to determine the current screen (e.g., Main Menu, In Battle).

2.  **`GameStateManager` (The Navigator)**
    * Takes screenshots of the screen.
    * Uses image templates (anchors) to identify the current game state.

3.  **`Vision` (The Eyes)**
    * If the bot is in a battle, the Vision module is activated.
    * It uses its sub-components (`EnemyDetector`, `CardClassifier`, `EasyOCR`) to analyze the screen and build a complete, structured understanding of the game state (tower health, your cards, elixir count, enemy positions).

4.  **`Agent` (The Brain)**
    * Receives the game state from the Vision module.
    * Feeds this state into its **Decision Transformer** model to decide the best action (which card to play and where).
    * After each game, it adds the entire match history to its `ReplayBuffer` and retrains its model.

5.  **`Controller` (The Hands)**
    * Receives the chosen action from the Agent.
    * Simulates mouse clicks to select the card and deploy it at the specified location on the screen.

***

## Setup and Installation

Follow these steps to get the bot running on your system.

### Prerequisites

* **Python 3.10+**
* An **NVIDIA GPU** with CUDA installed (required for the AI models).
* The game Clash Royale running in an Android emulator (developed and tested with **BlueStacks**).

### Installation Steps

1.  **Clone the Repository**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-folder>
    ```

2.  **Create a Python Virtual Environment**
    ```bash
    python -m venv venv
    ```

3.  **Activate the Virtual Environment**
    * On Windows (PowerShell):
        ```powershell
        .\venv\Scripts\Activate.ps1
        ```
    * On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

4.  **Install Dependencies**
    * The provided `requirements.txt` file contains all necessary packages, including the correct GPU-enabled version of PyTorch.
    ```bash
    pip install -r requirements.txt
    ```

***

## How to Run the Bot

Once the setup is complete, ensure the Clash Royale emulator is running and visible on your screen.

Then, simply run the main script from your activated virtual environment:
```bash
python runbot.py
```

The bot will find the game window, and you will see status messages printed in the console as it begins to play.

***

## Extending the Bot

### How to Add a New Card

To teach the bot to recognize and use a new card, follow these steps:

1.  **Collect Images**:
    * Use the `capture_cards.py` script to take hundreds of screenshots of the new card as it appears in your hand during gameplay.
    * Run the script and select the card slot(s) where the new card appears. Let it run for several games to collect varied images.

2.  **Organize the Dataset**:
    * Create a new folder inside `sorted_data/cards/`. The folder name must be the exact name of the new card (e.g., `sorted_data/cards/golem`).
    * Move all the screenshots you collected into this new folder.

3.  **Update the Game Configuration**:
    * Open the `config.py` file.
    * Add the new card's name and its elixir cost to the `CARD_COSTS` dictionary.
        ```python
        CARD_COSTS = {
            'archers': 3,
            # ... existing cards
            'golem': 8, # <-- Add the new card here
        }
        ```

4.  **Retrain the Card Classifier**:
    * Run the training script to update the `CardClassifier` model with the new card images.
        ```bash
        python train_classifier.py
        ```
    * This will process all the images in the `sorted_data/cards/` directory and save an updated `hand_classifier_best.pth` file. The bot will now recognize the new card.

***

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

***

## Recommended Usage

* **Consistency is Key**: Run the bot on the same emulator with the same resolution each time to ensure the vision system works reliably.
* **Initial Learning**: Let the bot play at least 50-100 games on its own. This will populate the `replay_buffer.pkl` file with a diverse set of experiences for it to learn from.
* **Dedicated Training**: While the bot learns a little after each game, its learning is shallow. For deep, meaningful improvement, periodically stop the bot and run a dedicated training session using the data it has collected.

***

## Troubleshooting

* **Error: "BlueStacks window not found"**: Make sure the emulator is running and that the window title in `scaler.py` exactly matches the title of your emulator window.
* **Error: `pygetwindow.PyGetWindowException`**: This is a permissions error. **Run your script as an Administrator**.
* **Error: `Activate.ps1 cannot be loaded...`**: This is a PowerShell security policy. Run this command in PowerShell once to fix it: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process`.

***

## Disclaimer

This project is for educational and research purposes only. Automating gameplay may be against the terms of service of some games. Use this software responsibly and at your own risk.

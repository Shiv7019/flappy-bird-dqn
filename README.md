#  Flappy Bird DQN — Reinforcement Learning Agent

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DQN-red)
![Gymnasium](https://img.shields.io/badge/Gymnasium-FlappyBird--v0-orange)

A from-scratch **Deep Q-Network (DQN)** agent that learns to play Flappy Bird through trial and error, built with **PyTorch** and **Gymnasium**. No one tells the bird the rules — it plays thousands of games, crashes a lot at first, and gradually teaches itself to flap at the right time.

This repo includes the training code, the network, the replay buffer, a pretrained model, and a script to play the game yourself.

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Training Results](#training-results)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Setup Guide — Windows 🪟](#setup-guide--windows-)
- [Setup Guide — macOS 🍎](#setup-guide--macos-)
- [How to Run](#how-to-run)
- [Configuring Hyperparameters](#configuring-hyperparameters)
- [Good to Know](#good-to-know)
- [Acknowledgments](#acknowledgments)

---

## Overview

This project trains an AI agent to play **Flappy Bird** using **Deep Q-Learning**, a reinforcement learning algorithm. At every moment in the game, the agent sees 12 numbers describing the bird and the upcoming pipes, and has to choose one of two actions: **flap** or **do nothing**. It gets rewarded for staying alive and passing pipes, and penalized for dying. Over thousands of episodes, a neural network learns which action to take in which situation, purely from experience.

**What's inside:**
-  Train a DQN agent from scratch
-  Watch your trained agent play automatically
-  Play the game yourself with the keyboard
-  All hyperparameters tweakable from one YAML file
-  A pretrained model + training log included, so you can see it in action immediately

---

## How It Works

In plain terms, here's the training loop:

1. **Observe** — the agent reads the current game state (bird position, velocity, distance/height of the next two pipes).
2. **Act** — a neural network (the "policy network") picks flap or don't-flap. Early in training it mostly picks *random* actions (exploration); as training goes on it relies more on what it has learned (exploitation). This trade-off is controlled by an **epsilon-greedy** strategy that decays over time.
3. **Get feedback** — the environment returns a reward (`+0.1` for surviving a frame, `+1` for passing a pipe, `-1` for dying) and the next state.
4. **Remember** — this experience `(state, action, next_state, reward, done)` is stored in a **replay memory** buffer.
5. **Learn** — every step, a random mini-batch of past experiences is sampled from memory and used to update the network with the Bellman equation, so learning isn't just based on the most recent moment.
6. **Stabilize** — a second, slower-updating "target network" is used when computing training targets, and is periodically synced with the policy network. This keeps training from chasing a constantly moving target.

This repeats for as many episodes as you let it run — the longer you train, the better the agent tends to get.

---

## Training Results

The included pretrained model (`flappybirdv0.pt`) was trained for **10,511 episodes**. Here's how its best score improved over that run (from `flappybirdv0.log`):

![Training Progress](assets/training_progress.png)

It went from crashing almost immediately (best reward ≈ **-8.7**) to reliably clearing several pipes (best reward ≈ **8.4**). Since the environment gives `+1` per pipe passed and `+0.1` per frame survived, a rising reward means the bird is both surviving longer *and* threading more pipes.

> 💡 DQN on Flappy Bird is a slow learner — don't be surprised if your own run needs several thousand episodes before you see it clear its first pipe consistently.

---

## Project Structure

```
├── agent.py                          # Main script — training & testing loop (start here)
├── dqn.py                            # The neural network (2-layer MLP): state → action values
├── experience_replay.py              # Replay memory (FIFO buffer of past experiences)
├── parameters.yaml                   # All hyperparameters, grouped by named "parameter set"
├── RL_FlappyBird.py                  # Play the game yourself with the spacebar
├── flappybirdv0.pt                   # Pretrained model weights (best agent found so far)
├── flappybirdv0.log                  # Training log matching the pretrained model
├── requirements.txt                  # Python dependencies (create this — see below)
└── runs/                             # Auto-created during training
    ├── flappybirdv0.pt               #   → best model gets saved here as training improves
    └── flappybirdv0.log              #   → new-best-reward log gets appended here
```

> **Note:** the code always saves/loads models from a `runs/` folder next to `agent.py` (see [Good to Know](#good-to-know) below for what this means for the pretrained model included here).

---

## Requirements

- **Python 3.10 or newer** (this project was built and tested on Python 3.12)
- `pip` (comes bundled with Python)
- No GPU required — it runs fine on CPU, just slower than GPU/MPS. `agent.py` automatically uses Apple Silicon (`mps`) or NVIDIA CUDA (`cuda`) if available, and falls back to `cpu` otherwise.

**Python packages used:**

| Package | What it's for |
|---|---|
| `gymnasium` | The base reinforcement-learning environment framework |
| `flappy-bird-gymnasium` | The actual Flappy Bird game, wrapped as a Gymnasium environment |
| `torch` | PyTorch — builds and trains the neural network |
| `pyyaml` | Reads `parameters.yaml` |
| `pygame` | Renders the game window and captures keyboard input |

---

## Setup Guide — Windows 

1. **Install Python**
   Download Python 3.12 (or newer) from [python.org/downloads](https://www.python.org/downloads/). During install, **check the box that says "Add python.exe to PATH"** — this saves a lot of headaches later.

2. **Install Git** *(optional, only if you want to `git clone` instead of downloading a ZIP)*
   Get it from [git-scm.com/download/win](https://git-scm.com/download/win) and install with the default options.

3. **Get the project onto your machine**
   Open **Command Prompt** or **PowerShell**, then run:
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```
   (Or download the repo as a ZIP from GitHub and extract it, then `cd` into that folder.)

4. **Create a virtual environment** (keeps this project's packages separate from the rest of your system)
   ```bash
   python -m venv venv
   ```

5. **Activate it**
   ```bash
   venv\Scripts\activate
   ```
   You should now see `(venv)` at the start of your terminal line.

6. **Install the dependencies**
   ```bash
   pip install gymnasium flappy-bird-gymnasium torch pyyaml pygame
   ```
   > If you have an NVIDIA GPU and want CUDA acceleration instead of CPU-only PyTorch, use the installer command from [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) instead of the plain `pip install torch` above.

7. **Run it!** See [How to Run](#how-to-run) below.

---

## Setup Guide — macOS 

1. **Install Python**
   Download Python 3.12 (or newer) from [python.org/downloads/macos](https://www.python.org/downloads/macos/), **or** if you use Homebrew:
   ```bash
   brew install python
   ```

2. **Install Git** *(optional, only if you want to `git clone` instead of downloading a ZIP)*
   macOS usually ships with Git already. Check with `git --version` in Terminal — if it's missing, it'll prompt you to install the Xcode Command Line Tools.

3. **Get the project onto your machine**
   Open **Terminal**, then run:
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```
   (Or download the repo as a ZIP from GitHub, extract it, then `cd` into that folder.)

4. **Create a virtual environment**
   ```bash
   python3 -m venv venv
   ```

5. **Activate it**
   ```bash
   source venv/bin/activate
   ```
   You should now see `(venv)` at the start of your terminal line.

6. **Install the dependencies**
   ```bash
   pip install gymnasium flappy-bird-gymnasium torch pyyaml pygame
   ```
   > On Apple Silicon Macs (M1/M2/M3/M4), PyTorch automatically uses the GPU via `mps` — no extra setup needed, `agent.py` detects it for you.
   >
   > If `pygame` fails to install, run `xcode-select --install` in Terminal to get the required build tools, then try again.

7. **Run it!** See [How to Run](#how-to-run) below.

---

## How to Run

All commands below assume your virtual environment is activated and you're inside the project folder.

###  Train a new agent from scratch

```bash
python agent.py flappybirdv0 --train
```

- `flappybirdv0` is the name of the hyperparameter set to use, as defined in `parameters.yaml`.
- Training runs **forever** by design (there's no automatic stop) — let it run as long as you like and stop it anytime with `Ctrl+C`. Every time the agent beats its previous best score, its weights are automatically saved to `runs/flappybirdv0.pt` and the improvement is logged to `runs/flappybirdv0.log`.
- Training runs without opening a game window (faster). To watch training happen live, you'd need to add `render=True` where `env.run()` is called — not required, just an option.

### 👀 Watch a trained agent play

```bash
python agent.py flappybirdv0
```

This loads the saved model from `runs/flappybirdv0.pt`, opens a game window, and lets the trained agent play on its own using what it has learned (no more randomness/exploration). See [Good to Know](#good-to-know) below for how to use the *included* pretrained model here.

### Play it yourself

```bash
python RL_FlappyBird.py
```

Opens a game window you control directly — press **Spacebar** to flap. Good for getting a feel for the game before (or after) watching the AI play it.

---

## Configuring Hyperparameters

All training settings live in `parameters.yaml`, grouped under a name (here, `flappybirdv0`) that you pass on the command line. Want to experiment? Duplicate the block, rename it, tweak values, and run `python agent.py <your_new_name> --train`.

| Parameter | Current Value | What it controls |
|---|---|---|
| `env_id` | `FlappyBird-v0` | Which Gymnasium environment to load |
| `alpha` | `0.001` | Learning rate for the optimizer |
| `gamma` | `0.99` | Discount factor — how much the agent values future rewards over immediate ones |
| `epsilon_init` | `1` | Starting exploration rate (1 = 100% random actions at first) |
| `epsilon_min` | `0.05` | The exploration rate never decays below this |
| `epsilon_decay` | `0.9995` | Multiplier applied to epsilon after every episode |
| `replay_memory_size` | `100000` | Max number of past experiences kept in memory |
| `mini_batch_size` | `32` | How many experiences are sampled per training update |
| `network_sync_rate` | `10` | How many steps between syncing the target network with the policy network |
| `reward_threshold` | `1000` | Safety cap — an episode ends early if this reward is reached |

---

## Good to Know

- **Using the included pretrained model:** the code always looks for models inside a `runs/` folder (i.e. `runs/flappybirdv0.pt`). If you want to immediately watch the pretrained agent play, create a `runs` folder in the project directory and move `flappybirdv0.pt` (and `flappybirdv0.log`, optionally) into it. If you just start training instead, that folder and those files get created automatically.
- **The `.pyc` files:** `dqn_cpython-312.pyc` and `experience_replay_cpython-312.pyc` are Python's *compiled bytecode cache* files, not source code — they're normally not committed to Git. Consider adding a `.gitignore` file with:
  ```
  __pycache__/
  *.pyc
  venv/
  ```
- **Training never stops on its own** — the episode loop runs until you interrupt it with `Ctrl+C`. That's expected behavior, not a bug.
- **Reward scale:** don't expect huge numbers. With `+0.1` per frame alive and `+1` per pipe passed, a "best reward" of `8` or `9` already means the bird is clearing several pipes in a row.

---

## Acknowledgments

- [flappy-bird-gymnasium](https://pypi.org/project/flappy-bird-gymnasium/) — the Gymnasium environment this project trains in
- [Gymnasium](https://gymnasium.farama.org/) (Farama Foundation) — the reinforcement learning environment framework
- [PyTorch](https://pytorch.org/) — for building and training the neural network

---

## License

No license file is included yet. If you're planning to make this repo public, consider adding one (e.g. [MIT](https://choosealicense.com/licenses/mit/) is a common, permissive choice for hobby ML projects) so others know how they can use your code.

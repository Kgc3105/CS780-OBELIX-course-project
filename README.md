# OBELIX Warehouse Robot — DDQN Agent
**CS780 Capstone Project | IIT Kanpur**  
**Author:** Kavuri Goutham Chandra (Roll No. 230553)

---

## Overview

This repository contains the trained reinforcement learning agent for the OBELIX warehouse robot task. The agent must locate, attach to, and push a grey box to the arena boundary using only an 18-bit binary sensor observation. The solution uses a **hybrid approach**: a Double Deep Q-Network (DDQN) combined with deterministic heuristic overrides for stuck recovery, box pushing, and forward-bias exploration.

---

## Files

```
agent.py          ← Main agent: policy, DDQN training, heuristics
ddqn_model.pth    ← Trained Q-network weights (auto-loaded on import)
ddqn_meta.npz     ← Training metadata: episode count, global step
```

> **Submission zip must contain:** `agent.py` + `ddqn_model.pth`

---

## Observation Space

The agent receives an **18-bit binary vector** at each step:

| Bits      | Count | Description                                      |
|-----------|-------|--------------------------------------------------|
| `0–3`     | 4     | Left sonar sensors (near + far)                  |
| `4–11`    | 8     | Front sonar sensors (near + far)                 |
| `12–15`   | 4     | Right sonar sensors (near + far)                 |
| `16`      | 1     | Infrared (IR) sensor — box directly ahead        |
| `17`      | 1     | Stuck flag — robot against wall or boundary      |

---

## Action Space

Five discrete actions:

| Action | Description         |
|--------|---------------------|
| `L45`  | Rotate left 45°     |
| `L22`  | Rotate left 22.5°   |
| `FW`   | Move forward        |
| `R22`  | Rotate right 22.5°  |
| `R45`  | Rotate right 45°    |

---

## Architecture

### Q-Network
```
Input  (18)  →  Linear → ReLU
Hidden (128) →  Linear → ReLU
Hidden (128) →  Linear
Output (5)       [one Q-value per action]
```

### Heuristic Override Layer (checked before querying network)

```
1. Stuck flag set?        → Execute escape queue (turn × N + FW × M)
2. IR bit active?         → Push-mode: FW if front clear, else turn toward open side
3. No override for > 4 steps? → Force FW (forward bias)
4. Training + ε-greedy?   → Random action with probability ε
5. Otherwise              → argmax Q(s, a) from network
```

---

## Reward Shaping (training only)

The environment reward is supplemented with shaped terms to accelerate learning:

| Condition                          | Shaped Reward                          |
|------------------------------------|----------------------------------------|
| Task success (`r ≥ 1999`)          | `+200`                                 |
| IR bit active (box attached)       | `+100`                                 |
| Ping-age penalty                   | `-0.13 × min(ping_age, 30)`            |
| Last action was `FW`               | `+2`                                   |
| Excess consecutive turns (> 2)     | `-5 × (consec_turns − 2)`              |
| Opposite turn oscillation          | `-5`                                   |

> Shaped rewards are used **only during training**. The evaluation score on Codabench uses the environment's native reward exclusively.

---

## Hyperparameters

| Parameter               | Value       |
|-------------------------|-------------|
| Learning rate (α)       | `5e-4`      |
| Discount factor (γ)     | `0.95`      |
| Batch size              | `64`        |
| Replay buffer size      | `50,000`    |
| Target update rate (τ)  | `0.005`     |
| ε start / end           | `1.0 / 0.05`|
| ε decay steps           | `50,000`    |
| Gradient clip norm      | `5.0`       |
| Optimizer               | Adam        |

---

## Training

### Requirements
```bash
pip install torch numpy
```

---

## Key Design Decisions

### Why pure learning failed
All purely learned approaches (DDQN, D3QN, VPG, A2C, DRQN) converged to spinning in place. The environment reward of `−1` per step with no directional gradient toward the box provides no useful learning signal until the agent accidentally contacts the box — an event that becomes increasingly unlikely as ε decays.

### Why heuristics first
The escape queue, push-mode, and forward-bias heuristics guarantee that the robot makes spatial progress and recovers from walls even under a random policy. Once these are in place, the replay buffer accumulates meaningful transitions and DDQN can learn a useful Q-function for the exploration phase.

### Escape queue logic
- **First stuck event:** turn toward the less-cluttered side (sonar count), then forward ×2.
- **Second+ stuck event on same side:** repeat the same turn 5 times before going forward — handles narrow wall corners.
- Free-step counter resets the stuck count after 4 unobstructed steps, preventing over-aggressive escaping in open space.

---

## Difficulty Levels

| Level | Setting              | Main Challenge                        |
|-------|----------------------|---------------------------------------|
| 1     | Static box           | Exploration, alignment                |
| 2     | Blinking box         | Signal loss, re-acquisition           |
| 3     | Moving + blinking    | Interception, pursuit under POMDP     |

All levels may include a vertical wall obstacle with a narrow opening.

---

## Competition Results (Codabench)

| Level                    | Rank |
|--------------------------|------|
| Level 1 — Static         | 11   |
| Level 2 — Blinking       | 24   |
| Level 3 — Moving+Blinking| 17   |
| **Test Phase (overall)** | **23** |

**Username:** `k_230553`

---

## Notes

- Training runs on **CPU only** — no GPU required.
- Model weights are loaded automatically when `agent.py` is imported.
- The `policy()` function signature matches the Codabench evaluation interface exactly.
- Do **not** call `reset_agent()` between steps of the same episode — only between episodes.

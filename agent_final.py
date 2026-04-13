from __future__ import annotations
from typing import Sequence, Optional
import os
import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")
_ACT_IDX = {a: i for i, a in enumerate(ACTIONS)}
_N_ACTS  = len(ACTIONS)

LR          = 5e-4
GAMMA       = 0.95
BATCH_SIZE  = 64
BUFFER_SIZE = 50000
TAU         = 0.005
EPS_START   = 1.0
EPS_END     = 0.05
EPS_DECAY   = 50000

_LEFT  = (0, 1, 2, 3)
_FRONT = (4, 5, 6, 7, 8, 9, 10, 11)
_RIGHT = (12, 13, 14, 15)
_IR    = 16
_STUCK = 17

_HERE       = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH = os.path.join(_HERE, "ddqn_model.pth")
_META_PATH  = os.path.join(_HERE, "ddqn_meta.npz")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):
    def __init__(self, input_dim: int = 18, output_dim: int = _N_ACTS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),       nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


q_net      = QNetwork().to(device)
target_net = QNetwork().to(device)
target_net.load_state_dict(q_net.state_dict())
target_net.eval()

optimizer = optim.Adam(q_net.parameters(), lr=LR)
replay_buffer = deque(maxlen=BUFFER_SIZE)


def load_model():
    if os.path.exists(_MODEL_PATH):
        q_net.load_state_dict(torch.load(_MODEL_PATH, map_location=device, weights_only=True))
        target_net.load_state_dict(q_net.state_dict())


def save_model():
    torch.save(q_net.state_dict(), _MODEL_PATH)
    np.savez(_META_PATH, episode=np.array(_st.episode), global_step=np.array(_st.global_step))


def reset_model_weights():
    global q_net, target_net, optimizer, replay_buffer
    q_net      = QNetwork().to(device)
    target_net = QNetwork().to(device)
    target_net.load_state_dict(q_net.state_dict())
    optimizer  = optim.Adam(q_net.parameters(), lr=LR)
    replay_buffer.clear()


load_model()


class _S:
    __slots__ = (
        "box_attached",
        "escape_queue",
        "last_action",
        "action_history",
        "ping_age",
        "consec_turns",
        "training",
        "episode",
        "global_step",
        "prev_obs",
        "prev_act_idx",
        "action_counter",
        "stuck_count",
        "last_turn",
        "free_steps"
    )

    def reset(self):
        self.box_attached = False
        self.escape_queue = []
        self.last_action  = "FW"
        self.action_history = []
        self.ping_age     = 30.0
        self.consec_turns = 0
        self.prev_obs     = None
        self.prev_act_idx = None
        self.action_counter = 0
        self.stuck_count  = 0
        self.last_turn    = None
        self.free_steps   = 0

    def new_episode(self, training: bool):
        self.reset()
        self.training = training
        if training:
            self.episode += 1


_st = _S()
_st.training = True

if os.path.exists(_META_PATH):
    meta = np.load(_META_PATH)
    _st.episode = int(meta["episode"])
    _st.global_step = int(meta.get("global_step", 0))
else:
    _st.episode = 0
    _st.global_step = 0

_st.reset()


def reset_agent(training: bool = True):
    _st.new_episode(training)


def _any_front(obs: np.ndarray) -> bool: return any(obs[i] for i in _FRONT)
def _any_left(obs: np.ndarray) -> bool:  return any(obs[i] for i in _LEFT)
def _any_sonar(obs: np.ndarray) -> bool: return any(obs[i] for i in range(16))

def _escape_turn(obs: np.ndarray) -> str:
    lc = sum(1 for i in _LEFT  if obs[i])
    rc = sum(1 for i in _RIGHT if obs[i])
    return "R45" if lc >= rc else "L45"


def _shape_reward(reward: float, ir: bool, obs: np.ndarray) -> float:
    if reward >= 1999: return 200.0
    if ir: return 100.0

    ping_penalty = -0.13 * min(_st.ping_age, 30.0)
    forward_reward = 2.0 if _st.last_action == "FW" else 0.0

    if _st.last_action in ("L45", "R45", "L22", "R22"):
        _st.consec_turns += 1
        turn_pen = -5.0 * max(0, _st.consec_turns - 2)
    else:
        _st.consec_turns = 0
        turn_pen = 0.0

    osc_pen = 0.0
    if len(_st.action_history) >= 2:
        curr_a = _st.action_history[-1]
        prev_a = _st.action_history[-2]
        if (prev_a == "L45" and curr_a == "R45") or \
           (prev_a == "R45" and curr_a == "L45") or \
           (prev_a == "L22" and curr_a == "R22") or \
           (prev_a == "R22" and curr_a == "L22"):
            osc_pen = -5.0

    return float(ping_penalty + forward_reward + turn_pen + osc_pen)


def ddqn_update():
    if len(replay_buffer) < BATCH_SIZE:
        return

    batch = random.sample(replay_buffer, BATCH_SIZE)
    state_b      = torch.FloatTensor(np.array([b[0] for b in batch])).to(device)
    action_b     = torch.LongTensor(np.array([b[1] for b in batch])).unsqueeze(1).to(device)
    reward_b     = torch.FloatTensor(np.array([b[2] for b in batch])).unsqueeze(1).to(device)
    next_state_b = torch.FloatTensor(np.array([b[3] for b in batch])).to(device)
    done_b       = torch.FloatTensor(np.array([b[4] for b in batch], dtype=np.float32)).unsqueeze(1).to(device)

    with torch.no_grad():
        next_action_idx = q_net(next_state_b).argmax(dim=1, keepdim=True)
        next_q_val = target_net(next_state_b).gather(1, next_action_idx)
        target_q   = reward_b + GAMMA * next_q_val * (1 - done_b)

    current_q = q_net(state_b).gather(1, action_b)

    loss = F.smooth_l1_loss(current_q, target_q)
    
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=5.0)
    optimizer.step()

    for target_param, local_param in zip(target_net.parameters(), q_net.parameters()):
        target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)


def _select_action(obs: np.ndarray, rng: np.random.Generator, forced_action: str = None):
    if forced_action is not None:
        return forced_action, _ACT_IDX[forced_action]

    if _st.training:
        epsilon = max(EPS_END, EPS_START - (_st.global_step / EPS_DECAY))
        if rng.random() < epsilon:
            act_idx = rng.integers(0, _N_ACTS)
            return ACTIONS[act_idx], act_idx

    with torch.no_grad():
        t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        q_vals = q_net(t)
        act_idx = q_vals.argmax(dim=1).item()
        return ACTIONS[act_idx], act_idx


def policy(obs: np.ndarray, rng: np.random.Generator, reward: float = 0.0, final_step: bool = False) -> str:
    ir    = bool(obs[_IR])
    stuck = bool(obs[_STUCK])
    any_s = _any_sonar(obs)

    if _st.training and _st.prev_obs is not None:
        shaped_r = _shape_reward(reward, ir, obs)
        replay_buffer.append((_st.prev_obs, _st.prev_act_idx, shaped_r, obs, final_step))
        ddqn_update()

    if final_step:
        return "FW"

    if _st.training:
        _st.global_step += 1
        
    _st.action_counter += 1

    if any_s and not stuck:
        _st.ping_age = 0.0
    else:
        _st.ping_age = min(_st.ping_age + 1.0, 30.0)

    if ir:
        _st.box_attached = True
    elif not ir and not stuck:
        _st.box_attached = False

    if stuck:
        _st.free_steps = 0
        if not _st.escape_queue or _st.last_action == "FW":
            _st.escape_queue.clear()
            _st.stuck_count += 1
            
            if _st.stuck_count >= 2 and _st.last_turn is not None:
                turn = _st.last_turn
                _st.escape_queue = [turn, turn, turn, turn, turn, "FW", "FW", "FW"]
            else:
                turn = _escape_turn(obs)
                _st.last_turn = turn
                if ir or _st.box_attached:
                    _st.escape_queue = [turn, turn, turn, turn, "FW", "FW"]
                else:
                    _st.escape_queue = [turn, turn, turn, "FW", "FW"]
    else:
        _st.free_steps += 1
        if _st.free_steps > 4:
            _st.stuck_count = 0

    forced_act = None

    if _st.escape_queue:
        forced_act = _st.escape_queue.pop(0)
        _st.action_counter = 0
    elif _st.box_attached:
        if ir and _any_front(obs):
            forced_act = "FW"
        elif ir:
            forced_act = "L22" if _any_left(obs) else "R22"
        else:
            forced_act = "L45" if _any_left(obs) else "R45"
        _st.action_counter = 0 
    elif _st.action_counter > 4:
        forced_act = "FW"
        _st.action_counter = 0

    action, act_idx = _select_action(obs, rng, forced_act)

    _st.prev_obs     = obs.copy()
    _st.prev_act_idx = act_idx
    _st.last_action  = action
    _st.action_history.append(action)
    
    if len(_st.action_history) > 4:
        _st.action_history.pop(0)

    return action
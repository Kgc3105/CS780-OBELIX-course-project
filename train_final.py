from __future__ import annotations
import argparse
import importlib.util
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent as _agent

def import_obelix(path: str):
    spec = importlib.util.spec_from_file_location("obelix_env", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX

def run_episode(OBELIX, seed, max_steps, difficulty, wall_obstacles,
                box_speed, scaling_factor, arena_size,
                training, render=False) -> dict:
    env = OBELIX(
        scaling_factor=scaling_factor, arena_size=arena_size,
        max_steps=max_steps, wall_obstacles=wall_obstacles,
        difficulty=difficulty, box_speed=box_speed, seed=seed,
    )
    obs = env.reset(seed=seed)
    _agent.reset_agent(training=training)

    rng          = np.random.default_rng(seed)
    total_reward = 0.0
    steps        = 0
    success      = False
    attached     = False
    wall_hits    = 0
    prev_reward  = 0.0

    for step in range(max_steps):
        action = _agent.policy(obs, rng, reward=prev_reward)
        obs, reward, done = env.step(action, render=render)
        prev_reward   = float(reward)
        total_reward += prev_reward
        steps         = step + 1

        if bool(obs[_agent._IR]) and not bool(obs[_agent._STUCK]):
            attached = True
        if bool(obs[_agent._STUCK]):
            wall_hits += 1
        if prev_reward >= 1999:
            success = True
        if done:
            break

    if training:
        _agent.policy(obs, rng, reward=prev_reward, final_step=True)

    return dict(reward=total_reward, steps=steps, success=success,
                attached=attached, wall_hits=wall_hits)

def _summary(label, rewards, successes, attached_l, steps_l):
    n = len(rewards)
    if not n:
        return
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Episodes    : {n}")
    print(f"  Success     : {sum(successes)}/{n}  ({100*np.mean(successes):.1f}%)")
    print(f"  Attach rate : {sum(attached_l)}/{n}  ({100*np.mean(attached_l):.1f}%)")
    print(f"  Mean reward : {np.mean(rewards):>10,.1f}")
    print(f"  Mean steps  : {np.mean(steps_l):.1f}")
    print(f"{'='*60}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py",      required=True)
    ap.add_argument("--train_eps",      type=int, default=2000)
    ap.add_argument("--eval_eps",       type=int, default=50)
    ap.add_argument("--max_steps",      type=int, default=2000)
    ap.add_argument("--difficulty",     type=int, default=3, choices=[0,1,2,3])
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--box_speed",      type=int, default=2)
    ap.add_argument("--scaling_factor", type=int, default=5)
    ap.add_argument("--arena_size",     type=int, default=500)
    ap.add_argument("--seed",           type=int, default=50)
    ap.add_argument("--render",         action="store_true")
    ap.add_argument("--render_every",   type=int, default=0)
    ap.add_argument("--log_every",      type=int, default=1)
    ap.add_argument("--save_every",     type=int, default=10)
    ap.add_argument("--resume",         action="store_true")
    args = ap.parse_args()

    if not args.resume and args.train_eps > 0:
        if os.path.exists(_agent._MODEL_PATH):
            os.remove(_agent._MODEL_PATH)
        if os.path.exists(_agent._META_PATH):
            os.remove(_agent._META_PATH)
        _agent.reset_model_weights()

    OBELIX = import_obelix(args.obelix_py)
    _DIFF  = {0: "static", 1: "static(p2)", 2: "blinking", 3: "moving+blinking"}

    ep_kw = dict(
        OBELIX=OBELIX, max_steps=args.max_steps, difficulty=args.difficulty,
        wall_obstacles=args.wall_obstacles, box_speed=args.box_speed,
        scaling_factor=args.scaling_factor, arena_size=args.arena_size,
    )

    print(f"{'='*60}")
    print(f"  OBELIX Agent — Flag-Based DDQN")
    print(f"{'='*60}")
    print(f"  Difficulty  : {args.difficulty} ({_DIFF[args.difficulty]})")
    print(f"  Train eps   : {args.train_eps}   Eval: {args.eval_eps}")
    print(f"  Resume      : {args.resume}")
    print(f"{'='*60}")

    tr, ts, ta, tst = [], [], [], []
    if args.train_eps > 0:
        for ep in range(args.train_eps):
            should_render = (
                args.render_every > 0 and (ep + 1) % args.render_every == 0
            )
            
            start_t = time.time()
            d = run_episode(seed=args.seed + ep, training=True, render=should_render, **ep_kw)
            ep_t = time.time() - start_t
            
            tr.append(d["reward"]); ts.append(int(d["success"]))
            ta.append(int(d["attached"])); tst.append(d["steps"])

            if (ep + 1) % args.log_every == 0:
                window = ts[-min(50, ep+1):]
                print(
                    f"Ep {ep+1:>4}/{args.train_eps}"
                    f"  r={d['reward']:>9.0f}"
                    f"  steps={d['steps']:>4}"
                    f"  walls={d['wall_hits']:>3}"
                    f"  time={ep_t:>5.2f}s"
                    f"  {'SUC' if d['success'] else 'ATT' if d['attached'] else '   '}"
                    f"  succ={100*np.mean(window):>5.1f}%"
                )

            if (ep + 1) % args.save_every == 0:
                _agent.save_model()

        _agent.save_model()
        _summary("TRAINING SUMMARY", tr, ts, ta, tst)

    if args.eval_eps > 0:
        print(f"\n── EVALUATION ({args.eval_eps} eps, greedy) ──")
        er, es, ea, est = [], [], [], []
        for ep in range(args.eval_eps):
            start_t = time.time()
            d = run_episode(seed=args.seed + 10000 + ep, training=False,
                            render=args.render, **ep_kw)
            ep_t = time.time() - start_t
            
            er.append(d["reward"]); es.append(int(d["success"]))
            ea.append(int(d["attached"])); est.append(d["steps"])

            if (ep + 1) % args.log_every == 0:
                window = es[-min(50, ep+1):]
                print(
                    f"[EVAL] Ep {ep+1:>4}/{args.eval_eps}"
                    f"  r={d['reward']:>9.0f}"
                    f"  steps={d['steps']:>4}"
                    f"  walls={d['wall_hits']:>3}"
                    f"  time={ep_t:>5.2f}s"
                    f"  {'SUC' if d['success'] else 'ATT' if d['attached'] else '   '}"
                    f"  succ={100*np.mean(window):>5.1f}%"
                )

        _summary("EVALUATION RESULTS", er, es, ea, est)

if __name__ == "__main__":
    main()
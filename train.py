import argparse
import logging
import multiprocessing as mp
import os
import signal
import sys
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Bernoulli
from tqdm import tqdm

from ai import ConvSNN, HIDDEN_SIZE
from game import GameEnv, get_stage_spec

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("train.log", mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

FRAME_SKIP = 2
NUM_ENVS = 128
GAMMA = 0.990
GAE_LAMBDA = 0.95
PPO_EPOCHS = 4
CLIP_EPS = 0.2
VALUE_COEF = 0.5
ENT_START = 0.05
ENT_END = 0.02
LR = 1.5e-4
MAX_GRAD = 0.5
TOTAL_EPS = 50000
ROLLING_WIN_WINDOW = 200
SAVE_EVERY = 2500
STOP_SIGNAL_FILE = "STOP_AND_SAVE"

# MANUAL_STOP = False

MANUAL_STOP = False


def _sigint_handler(_signum, _frame):
    global MANUAL_STOP
    MANUAL_STOP = True


signal.signal(signal.SIGINT, _sigint_handler)


def resolve_stage(resume_stage, forced_stage=None):
    if forced_stage is not None:
        return forced_stage
    return resume_stage


def compute_gae(rewards, values, gamma=GAMMA, lam=GAE_LAMBDA):
    advantages = []
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_v = values[t + 1] if t < len(rewards) - 1 else 0.0
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    return advantages


def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _worker(remote, parent_remote, env_fn, stage_id):
    parent_remote.close()
    env = env_fn(stage_id)

    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                act, enemy_act, frame_skip = data
                state, reward, done, info = env.step(act, enemy_ai_action=enemy_act, frame_skip=frame_skip)
                if done:
                    new_state = env.reset()
                    remote.send((state, reward, True, new_state, info))
                else:
                    remote.send((state, reward, False, None, info))
            elif cmd == "reset":
                remote.send(env.reset())
            elif cmd == "set_stage":
                env.set_stage(data)
                remote.send(True)
            elif cmd == "close":
                remote.close()
                break
            else:
                raise NotImplementedError(cmd)
    except KeyboardInterrupt:
        pass
    finally:
        if getattr(env, "render_mode", False):
            import pygame

            pygame.quit()


class VecEnv:
    def __init__(self, env_fn, num_envs, stage_id):
        self.num_envs = num_envs
        self.waiting = False
        self.closed = False
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(num_envs)])
        self.procs = []

        for wr, r in zip(self.work_remotes, self.remotes):
            p = mp.Process(target=_worker, args=(wr, r, env_fn, stage_id), daemon=True)
            p.start()
            self.procs.append(p)
            wr.close()

    def reset(self):
        for r in self.remotes:
            r.send(("reset", None))
        return [r.recv() for r in self.remotes]

    def set_stage(self, stage_id):
        for r in self.remotes:
            r.send(("set_stage", stage_id))
        for r in self.remotes:
            _ = r.recv()

    def step_async(self, actions, enemy_actions, frame_skip):
        for i, r in enumerate(self.remotes):
            r.send(("step", (actions[i], enemy_actions[i], frame_skip)))
        self.waiting = True

    def step_wait(self):
        results = [r.recv() for r in self.remotes]
        self.waiting = False
        states, rewards, dones, new_states, infos = zip(*results)
        return list(states), list(rewards), list(dones), list(new_states), list(infos)

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for r in self.remotes:
                r.recv()
        for r in self.remotes:
            r.send(("close", None))
        for p in self.procs:
            p.join()
        self.closed = True


def save_checkpoint(path, model, optimizer, total_eps_done, stage_id):
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "total_eps_done": total_eps_done,
        "stage_id": stage_id,
    }
    torch.save(payload, path)


def load_checkpoint(path, model, optimizer, device):
    ckpt = torch.load(path, map_location=device)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
        if optimizer is not None and "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        return int(ckpt.get("total_eps_done", 0)), int(ckpt.get("stage_id", 0))

    # backward compatibility: pure state_dict
    model.load_state_dict(ckpt)
    return 0, 0


def train(resume_path=None, forced_stage=None, target_stage_eps=50000):
    global MANUAL_STOP

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"訓練裝置: {device}")

    model = ConvSNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    total_eps_done = 0
    save_milestone = SAVE_EVERY
    resume_stage = 0

    if resume_path and os.path.exists(resume_path):
        logger.info(f"載入 checkpoint: {resume_path}")
        ckpt_eps_done, resume_stage = load_checkpoint(resume_path, model, optimizer, device)
        if forced_stage is not None and forced_stage != resume_stage:
            logger.info("進入新階段，保留模型權重，局數歸零")
            total_eps_done = 0
        else:
            total_eps_done = ckpt_eps_done
        save_milestone = ((total_eps_done // SAVE_EVERY) + 1) * SAVE_EVERY
        logger.info(f"續訓起點: episodes={total_eps_done}, stage={resume_stage}, next_save={save_milestone}")

    current_stage = forced_stage if forced_stage is not None else resume_stage

    logger.info(f"啟動 {NUM_ENVS} 個環境, 初始階段 Stage {current_stage} - {get_stage_spec(current_stage).name}")
    vec_env = VecEnv(env_fn=lambda sid: GameEnv(render_mode=False, stage_id=sid), num_envs=NUM_ENVS, stage_id=current_stage)

    batch_episodes = max(1, target_stage_eps // NUM_ENVS)
    start_batch = total_eps_done // NUM_ENVS
    start_time = time.time()

    rolling_win = deque(maxlen=ROLLING_WIN_WINDOW)

    pbar = tqdm(range(start_batch, batch_episodes), desc="PPO", unit="batch", dynamic_ncols=True)

    for batch_ep in pbar:
        if MANUAL_STOP or os.path.exists(STOP_SIGNAL_FILE):
            stop_path = f"snn_S{current_stage}_ep_{total_eps_done}_manual_stop.pth"
            save_checkpoint(stop_path, model, optimizer, total_eps_done, current_stage)
            logger.info(f"收到中斷指令，已存檔並停止: {stop_path}")
            if os.path.exists(STOP_SIGNAL_FILE):
                os.remove(STOP_SIGNAL_FILE)
            break

        batch_start_time = time.time()

        # remove automatic change current_stage
        vec_env.set_stage(current_stage)
        stage_spec = get_stage_spec(current_stage)

        progress = total_eps_done / max(1, target_stage_eps)
        entropy_coef = ENT_START + (ENT_END - ENT_START) * progress

        model.eval()
        h_ai = torch.zeros(1, NUM_ENVS, HIDDEN_SIZE, device=device)
        h_enemy = torch.zeros(1, NUM_ENVS, HIDDEN_SIZE, device=device)

        states_list = vec_env.reset()

        traj = [{"states": [], "actions": [], "rewards": [], "values": [], "old_lp": []} for _ in range(NUM_ENVS)]
        episode_done = [False] * NUM_ENVS
        env_kills = [0] * NUM_ENVS
        env_wins = [0] * NUM_ENVS

        while not all(episode_done):
            s_np = np.stack([s[0] for s in states_list])
            sc_np = np.stack([s[1] for s in states_list])
            s_batch = torch.as_tensor(s_np, dtype=torch.float32, device=device)
            sc_batch = torch.as_tensor(sc_np, dtype=torch.float32, device=device)

            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits, values, h_ai_new = model(s_batch, sc_batch, h_ai)

                logits = logits.float()
                values = values.float().squeeze(-1)
                dist = Bernoulli(torch.sigmoid(logits))
                acts = dist.sample()
                lp = dist.log_prob(acts).sum(dim=1)

                # stage5 自我博弈：敵人也用同一模型
                if current_stage == 5:
                    logits_e, _, h_enemy_new = model(s_batch, sc_batch, h_enemy)
                    dist_e = Bernoulli(torch.sigmoid(logits_e.float()))
                    enemy_acts = dist_e.sample()
                    h_enemy = h_enemy_new.clone()
                else:
                    enemy_acts = torch.zeros_like(acts)

            h_ai = h_ai_new.clone()

            acts_np = acts.cpu().numpy()
            enemy_acts_np = enemy_acts.cpu().numpy()
            vals_np = values.cpu().numpy()
            lp_np = lp.cpu().numpy()

            for i in range(NUM_ENVS):
                if episode_done[i]:
                    continue
                traj[i]["states"].append((torch.from_numpy(s_np[i]), torch.from_numpy(sc_np[i])))
                traj[i]["actions"].append(torch.from_numpy(acts_np[i]))
                traj[i]["values"].append(float(vals_np[i]))
                traj[i]["old_lp"].append(float(lp_np[i]))

            vec_env.step_async(acts_np, enemy_acts_np, FRAME_SKIP)
            prev_states, rewards, dones, new_states, infos = vec_env.step_wait()

            next_states = list(states_list)
            for i in range(NUM_ENVS):
                if episode_done[i]:
                    continue
                traj[i]["rewards"].append(float(rewards[i]))
                if dones[i]:
                    episode_done[i] = True
                    next_states[i] = new_states[i]
                    h_ai[:, i, :] = 0.0
                    h_enemy[:, i, :] = 0.0
                    env_kills[i] = infos[i].get("kill_count", 0)
                    env_wins[i] = 1 if infos[i].get("ai_win", False) else 0
                else:
                    next_states[i] = prev_states[i]
            states_list = next_states

        all_advs = []
        avg_rewards = []
        for i in range(NUM_ENVS):
            rews = traj[i]["rewards"]
            vals = traj[i]["values"]
            avg_rewards.append(sum(rews) if rews else 0.0)
            if not rews:
                traj[i]["advantages"] = []
                traj[i]["returns"] = []
                continue
            advs = compute_gae(rews, vals)
            rets = [a + v for a, v in zip(advs, vals)]
            traj[i]["advantages"] = advs
            traj[i]["returns"] = rets
            all_advs.extend(advs)

        if len(all_advs) > 1:
            a_mean = float(np.mean(all_advs))
            a_std = float(np.std(all_advs)) + 1e-8
            for i in range(NUM_ENVS):
                traj[i]["advantages"] = [(a - a_mean) / a_std for a in traj[i]["advantages"]]

        t_per_env = [len(traj[i]["states"]) for i in range(NUM_ENVS)]
        max_t = max(t_per_env) if max(t_per_env) > 0 else 1
        s_shape = (4, 15, 15)

        bat_states = torch.zeros(max_t, NUM_ENVS, *s_shape)
        bat_scalars = torch.zeros(max_t, NUM_ENVS, 9)
        bat_actions = torch.zeros(max_t, NUM_ENVS, 9)
        bat_old_lp = torch.zeros(max_t, NUM_ENVS)
        bat_adv = torch.zeros(max_t, NUM_ENVS)
        bat_ret = torch.zeros(max_t, NUM_ENVS)
        mask = torch.zeros(max_t, NUM_ENVS, dtype=torch.bool)

        for i in range(NUM_ENVS):
            t_i = t_per_env[i]
            if t_i == 0:
                continue
            mask[:t_i, i] = True
            bat_states[:t_i, i] = torch.stack([x[0] for x in traj[i]["states"]])
            bat_scalars[:t_i, i] = torch.stack([x[1] for x in traj[i]["states"]])
            bat_actions[:t_i, i] = torch.stack(traj[i]["actions"])
            bat_old_lp[:t_i, i] = torch.tensor(traj[i]["old_lp"])
            bat_adv[:t_i, i] = torch.tensor(traj[i]["advantages"])
            bat_ret[:t_i, i] = torch.tensor(traj[i]["returns"])

        bat_states = bat_states.to(device)
        bat_scalars = bat_scalars.to(device)
        bat_actions = bat_actions.to(device)
        bat_old_lp = bat_old_lp.to(device)
        bat_adv = bat_adv.to(device)
        bat_ret = bat_ret.to(device)
        mask = mask.to(device)

        model.train()
        for _ in range(PPO_EPOCHS):
            optimizer.zero_grad()
            h = torch.zeros(1, NUM_ENVS, HIDDEN_SIZE, device=device)
            ep_losses = []

            for t in range(max_t):
                s = bat_states[t]
                sc = bat_scalars[t]
                a = bat_actions[t]
                lp_old = bat_old_lp[t]
                adv = bat_adv[t]
                ret = bat_ret[t]
                m_t = mask[t]

                if not m_t.any():
                    with torch.no_grad():
                        _, _, h = model(s, sc, h)
                    h = h.detach()
                    continue

                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits, val, h = model(s, sc, h)

                logits = logits.float()
                val = val.float().squeeze(-1)
                probs = torch.sigmoid(logits)
                dist = Bernoulli(probs)
                new_lp = dist.log_prob(a).sum(dim=1)
                entropy = dist.entropy().sum(dim=1)

                ratio = torch.exp(new_lp - lp_old)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv
                actor_l = -torch.min(surr1, surr2)
                critic_l = (val - ret).pow(2)

                valid = m_t.float()
                n_valid = valid.sum().clamp(min=1)
                step_loss = actor_l + VALUE_COEF * critic_l - entropy_coef * entropy
                t_loss = (step_loss * valid).sum() / n_valid
                ep_losses.append(t_loss)

            if ep_losses:
                total_loss = torch.stack(ep_losses).mean()
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD)
                scaler.step(optimizer)
                scaler.update()

        total_eps_done += NUM_ENVS
        batch_time = time.time() - batch_start_time
        elapsed = time.time() - start_time

        avg_rew = float(np.mean(avg_rewards))

        for k, w in zip(env_kills, env_wins):
            rolling_win.append((k, w))
        
        # 計算統計資料
        roll_list = list(rolling_win)
        total_ep = len(roll_list)
        dist_str = ""
        rolling_win_rate = 0.0
        
        if total_ep > 0:
            rolling_win_rate = sum(x[1] for x in roll_list) / total_ep
            
            # 只有追殺 NPC 模式 (scripted 模式) 才顯示擊殺分佈
            if stage_spec.mode == "scripted":
                max_e = stage_spec.enemy_count
                kills_only = [x[0] for x in roll_list]
                dist = [kills_only.count(i) / total_ep for i in range(max_e + 1)]
                dist_str = " ".join([f"k{i}:{p:.2f}" for i, p in enumerate(dist)])
        
        show_stats = current_stage <= 6

        batches_done = batch_ep - start_batch + 1
        total_batches = batch_episodes - start_batch
        eta_sec = (elapsed / max(1, batches_done)) * max(0, total_batches - batches_done)

        postfix = {
            "stage": f"S{current_stage}",
            "eps": f"{total_eps_done}",
            "rew": f"{avg_rew:.1f}",
            "ent": f"{entropy_coef:.3f}",
        }
        if show_stats:
            postfix["win"] = f"{rolling_win_rate:.3f}"
            if dist_str:
                postfix["kills"] = dist_str
        pbar.set_postfix(postfix)

        msg = (
            f"進度: Stage{current_stage}-{stage_spec.name}, eps={total_eps_done}, "
            f"avg_rew={avg_rew:.2f}, ent={entropy_coef:.3f}"
        )
        if show_stats:
            msg += f", win_rate={rolling_win_rate:.3f}"
            if dist_str:
                msg += f", kills=[{dist_str}]"
        msg += f" | batch={batch_time:.1f}s, elapsed={format_time(elapsed)}, ETA={format_time(eta_sec)}"
        logger.info(msg)

        if total_eps_done >= save_milestone:
            path = f"snn_S{current_stage}_ep_{save_milestone}.pth"
            save_checkpoint(path, model, optimizer, total_eps_done, current_stage)
            logger.info(f"已儲存 checkpoint: {path}")
            save_milestone += SAVE_EVERY

    pbar.close()
    vec_env.close()

    final_path = f"snn_S{current_stage}_ep_final.pth"
    save_checkpoint(final_path, model, optimizer, total_eps_done, current_stage)
    logger.info(f"訓練結束，已儲存: {final_path}")
    logger.info(f"中斷存檔指令: touch {STOP_SIGNAL_FILE}")


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="checkpoint path")
    parser.add_argument("--stage", type=int, default=None, choices=[0, 1, 2, 3, 4, 5, 6], help="固定訓練階段")
    parser.add_argument("--stage_eps", type=int, default=50000, help="本階段訓練局數")
    args = parser.parse_args()

    train(resume_path=args.resume, forced_stage=args.stage, target_stage_eps=args.stage_eps)

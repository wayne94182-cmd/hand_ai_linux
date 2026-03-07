"""
train.py — MAPPO + N-Agent Flatten 訓練腳本
支援多個 learning agents、LSTM 隱藏狀態管理、
通訊向量、action masking、以及分離的 Actor / Critic 優化。
修正：feat.detach / action masking / comm team filter / critic team grouping
"""
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
from torch.distributions import Bernoulli, Normal
from tqdm import tqdm

from ai import ConvLSTM, TeamPoolingCritic, CommHandler, HIDDEN_SIZE, NUM_COMM
from game import GameEnv, get_stage_spec
from game.env import NUM_CHANNELS, NUM_SCALARS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("train.log", mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ── 超參數 ──────────────────────────────────────────────
FRAME_SKIP = 2
NUM_ENVS = 64
GAMMA = 0.990
GAE_LAMBDA = 0.95
PPO_EPOCHS = 4
CLIP_EPS = 0.2
VALUE_COEF = 0.5
ENT_START = 0.1
ENT_END = 0.03
LR = 3e-4
MAX_GRAD = 0.5
TOTAL_EPS = 50000
ROLLING_WIN_WINDOW = 200
SAVE_EVERY = 2500
STOP_SIGNAL_FILE = "STOP_AND_SAVE"

# 新增超參數
COMM_ENT_COEF = 0.01
INDIVIDUAL_REWARD_WEIGHT = 0.6
TEAM_REWARD_WEIGHT = 0.4

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


# ═══════════════════════════════════════════════════════
#  VecEnv Worker（支援多 AI）
# ═══════════════════════════════════════════════════════

def _worker(remote, parent_remote, env_fn, stage_id, n_ai):
    parent_remote.close()
    env = env_fn(stage_id, n_ai)

    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                actions_list, frame_skip = data
                result = env.step(actions_list, frame_skip=frame_skip)
                if n_ai == 1:
                    state, reward, done, info = result
                    states_list = [state]
                    rewards_list = [reward]
                else:
                    states_list, rewards_list, done, info = result

                if done:
                    new_result = env.reset()
                    if n_ai == 1:
                        new_states_list = [new_result]
                    else:
                        new_states_list = new_result
                    remote.send((states_list, rewards_list, True, new_states_list, info))
                else:
                    remote.send((states_list, rewards_list, False, None, info))
            elif cmd == "reset":
                result = env.reset()
                if n_ai == 1:
                    remote.send([result])
                else:
                    remote.send(result)
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
    def __init__(self, env_fn, num_envs, stage_id, n_ai):
        self.num_envs = num_envs
        self.n_ai = n_ai
        self.waiting = False
        self.closed = False
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(num_envs)])
        self.procs = []

        for wr, r in zip(self.work_remotes, self.remotes):
            p = mp.Process(target=_worker, args=(wr, r, env_fn, stage_id, n_ai), daemon=True)
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

    def step_async(self, actions_per_env, frame_skip):
        for i, r in enumerate(self.remotes):
            r.send(("step", (actions_per_env[i], frame_skip)))
        self.waiting = True

    def step_wait(self):
        results = [r.recv() for r in self.remotes]
        self.waiting = False
        states_list, rewards_list, dones, new_states_list, infos = zip(*results)
        return (list(states_list), list(rewards_list),
                list(dones), list(new_states_list), list(infos))

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


# ═══════════════════════════════════════════════════════
#  Checkpoint
# ═══════════════════════════════════════════════════════

def save_checkpoint(path, model, critic, optimizer, optimizer_critic,
                    total_eps_done, stage_id, n_ai):
    payload = {
        "model_state": model.state_dict(),
        "critic_state": critic.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "optimizer_critic_state": optimizer_critic.state_dict(),
        "total_eps_done": total_eps_done,
        "stage_id": stage_id,
        "n_ai": n_ai,
    }
    torch.save(payload, path)


def load_checkpoint(path, model, critic, optimizer, optimizer_critic, device):
    ckpt = torch.load(path, map_location=device)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
        if optimizer is not None and "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if critic is not None and "critic_state" in ckpt:
            critic.load_state_dict(ckpt["critic_state"])
        if optimizer_critic is not None and "optimizer_critic_state" in ckpt:
            optimizer_critic.load_state_dict(ckpt["optimizer_critic_state"])
        return (int(ckpt.get("total_eps_done", 0)),
                int(ckpt.get("stage_id", 0)),
                int(ckpt.get("n_ai", 1)))

    # backward compatibility: pure state_dict
    model.load_state_dict(ckpt)
    return 0, 0, 1


# ═══════════════════════════════════════════════════════
#  主訓練迴圈
# ═══════════════════════════════════════════════════════

def train(resume_path=None, forced_stage=None, target_stage_eps=50000, n_ai=2):
    global MANUAL_STOP

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"訓練裝置: {device}, N_AI={n_ai}")

    FLAT_BATCH = n_ai * NUM_ENVS

    model = ConvLSTM().to(device)
    critic = TeamPoolingCritic().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    optimizer_critic = optim.Adam(critic.parameters(), lr=LR)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    total_eps_done = 0
    save_milestone = SAVE_EVERY
    resume_stage = 0

    if resume_path and os.path.exists(resume_path):
        logger.info(f"載入 checkpoint: {resume_path}")
        ckpt_eps, resume_stage, ckpt_n_ai = load_checkpoint(
            resume_path, model, critic, optimizer, optimizer_critic, device)
        if forced_stage is not None and forced_stage != resume_stage:
            logger.info("進入新階段，保留模型權重，局數歸零")
            total_eps_done = 0
        else:
            total_eps_done = ckpt_eps
        save_milestone = ((total_eps_done // SAVE_EVERY) + 1) * SAVE_EVERY
        logger.info(f"續訓起點: episodes={total_eps_done}, stage={resume_stage}, "
                    f"ckpt_n_ai={ckpt_n_ai}, next_save={save_milestone}")

    current_stage = forced_stage if forced_stage is not None else resume_stage

    logger.info(f"啟動 {NUM_ENVS} 個環境 × {n_ai} AI, "
                f"初始階段 Stage {current_stage} - {get_stage_spec(current_stage).name}")

    vec_env = VecEnv(
        env_fn=lambda sid, nai: GameEnv(render_mode=False, stage_id=sid,
                                        n_learning_agents=nai),
        num_envs=NUM_ENVS,
        stage_id=current_stage,
        n_ai=n_ai,
    )

    batch_episodes = max(1, target_stage_eps // NUM_ENVS)
    start_batch = total_eps_done // NUM_ENVS
    start_time = time.time()

    rolling_win = deque(maxlen=ROLLING_WIN_WINDOW)

    pbar = tqdm(range(start_batch, batch_episodes), desc="PPO", unit="batch",
                dynamic_ncols=True)

    for batch_ep in pbar:
        if MANUAL_STOP or os.path.exists(STOP_SIGNAL_FILE):
            stop_path = f"snn_S{current_stage}_ep_{total_eps_done}_manual_stop.pth"
            save_checkpoint(stop_path, model, critic, optimizer, optimizer_critic,
                            total_eps_done, current_stage, n_ai)
            logger.info(f"收到中斷指令，已存檔並停止: {stop_path}")
            if os.path.exists(STOP_SIGNAL_FILE):
                os.remove(STOP_SIGNAL_FILE)
            break

        batch_start_time = time.time()

        vec_env.set_stage(current_stage)
        stage_spec = get_stage_spec(current_stage)

        progress = total_eps_done / max(1, target_stage_eps)
        entropy_coef = ENT_START + (ENT_END - ENT_START) * progress

        model.eval()

        # 隱藏狀態：展平 (n_ai × NUM_ENVS)
        h = torch.zeros(1, FLAT_BATCH, HIDDEN_SIZE, device=device)
        c = torch.zeros(1, FLAT_BATCH, HIDDEN_SIZE, device=device)
        last_comm = np.zeros((FLAT_BATCH, NUM_COMM), dtype=np.float32)

        # 重置所有環境
        env_states = vec_env.reset()
        # env_states[j] = list of n_ai (view, scalar, team_id) tuples

        # 追蹤每個 flat index 目前的 action mask
        last_masks = np.ones((FLAT_BATCH, 12), dtype=bool)

        # 軌跡收集（per flat index）
        traj = [{"states": [], "actions": [], "rewards": [], "values": [],
                 "old_lp": [], "comm_acts": [], "comm_lp": [],
                 "comm_mu": [], "comm_logstd": [], "masks": []}
                for _ in range(FLAT_BATCH)]
        episode_done = [False] * NUM_ENVS
        env_kills = [0] * NUM_ENVS
        env_wins = [0] * NUM_ENVS

        while not all(episode_done):
            # 1. 從 NUM_ENVS 個環境收集狀態
            s_np = np.zeros((NUM_ENVS, n_ai, NUM_CHANNELS, 15, 15), dtype=np.float32)
            sc_np = np.zeros((NUM_ENVS, n_ai, NUM_SCALARS), dtype=np.float32)
            team_ids = np.zeros((NUM_ENVS, n_ai), dtype=np.int32)
            for j in range(NUM_ENVS):
                for i in range(n_ai):
                    s_np[j, i] = env_states[j][i][0]
                    sc_np[j, i] = env_states[j][i][1]
                    team_ids[j, i] = int(env_states[j][i][2])

            # 2. 展平：先 AI index，再 env index
            s_flat = s_np.transpose(1, 0, 2, 3, 4).reshape(FLAT_BATCH, NUM_CHANNELS, 15, 15)
            sc_flat = sc_np.transpose(1, 0, 2).reshape(FLAT_BATCH, NUM_SCALARS)

            s_t = torch.as_tensor(s_flat, dtype=torch.float32, device=device)
            sc_t = torch.as_tensor(sc_flat, dtype=torch.float32, device=device)

            # 3. 準備 comm_in — 只傳同 env、同 team、不同 agent (Fix 5)
            K = max(0, n_ai - 1)
            comm_in_np = np.zeros((FLAT_BATCH, K, NUM_COMM), dtype=np.float32)
            for i in range(n_ai):
                for j in range(NUM_ENVS):
                    flat_idx = i * NUM_ENVS + j
                    my_team = team_ids[j, i]
                    k_idx = 0
                    for i2 in range(n_ai):
                        if i2 == i:
                            continue
                        if team_ids[j, i2] != my_team:  # 不同 team 跳過
                            continue
                        flat_other = i2 * NUM_ENVS + j
                        if k_idx < K:
                            comm_in_np[flat_idx, k_idx] = last_comm[flat_other]
                            k_idx += 1

            comm_in_t = torch.as_tensor(comm_in_np, dtype=torch.float32, device=device)

            # 4. 前向傳播
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits, mu, logstd, feat, (h_new, c_new) = model(
                        s_t, sc_t, (h, c), comm_in_t)

                logits = logits.float()
                feat = feat.float()

                # Fix 1: Critic 使用 feat.detach()
                v = critic(feat.detach())  # (FLAT_BATCH,)

                # Fix 2: Action Masking 在 rollout 取樣時
                mask_t = torch.as_tensor(last_masks, dtype=torch.bool, device=device)
                logits_masked = logits.masked_fill(~mask_t, -1e9)
                dist_disc = Bernoulli(torch.sigmoid(logits_masked))
                acts = dist_disc.sample()
                lp_disc = dist_disc.log_prob(acts).sum(dim=1)

                comm_vec, lp_comm = CommHandler.sample(mu, logstd)
                comm_np_new = CommHandler.to_env(comm_vec)

            h = h_new.clone()
            c = c_new.clone()

            acts_np = acts.cpu().numpy()
            vals_np = v.cpu().numpy()
            lp_disc_np = lp_disc.cpu().numpy()
            lp_comm_np = lp_comm.cpu().numpy()
            mu_np = mu.cpu().numpy()
            logstd_np = logstd.cpu().numpy()

            # 存軌跡
            for i in range(n_ai):
                for j in range(NUM_ENVS):
                    if episode_done[j]:
                        continue
                    flat = i * NUM_ENVS + j
                    traj[flat]["states"].append((
                        torch.from_numpy(s_flat[flat].copy()),
                        torch.from_numpy(sc_flat[flat].copy()),
                    ))
                    traj[flat]["actions"].append(torch.from_numpy(acts_np[flat].copy()))
                    traj[flat]["values"].append(float(vals_np[flat]))
                    traj[flat]["old_lp"].append(float(lp_disc_np[flat]))
                    traj[flat]["comm_acts"].append(
                        torch.from_numpy(comm_np_new[flat].copy()))
                    traj[flat]["comm_lp"].append(float(lp_comm_np[flat]))
                    traj[flat]["comm_mu"].append(
                        torch.from_numpy(mu_np[flat].copy()))
                    traj[flat]["comm_logstd"].append(
                        torch.from_numpy(logstd_np[flat].copy()))
                    traj[flat]["masks"].append(
                        torch.from_numpy(last_masks[flat].copy()))

            # 6. 還原形狀派發給 env
            acts_env = acts_np.reshape(n_ai, NUM_ENVS, 12).transpose(1, 0, 2)
            last_comm = comm_np_new.copy()

            env_actions = []
            for j in range(NUM_ENVS):
                env_actions.append([acts_env[j][i].tolist() for i in range(n_ai)])

            vec_env.step_async(env_actions, FRAME_SKIP)
            all_states, all_rewards, dones, new_all_states, infos = vec_env.step_wait()

            next_env_states = list(env_states)
            for j in range(NUM_ENVS):
                if episode_done[j]:
                    continue
                # 存 reward
                for i in range(n_ai):
                    flat = i * NUM_ENVS + j
                    rew = all_rewards[j][i] if isinstance(all_rewards[j], list) else all_rewards[j]
                    traj[flat]["rewards"].append(float(rew))

                # Fix 2B: 從 info 取回真實 action mask
                raw_masks = infos[j].get("action_masks", [[True]*12]*n_ai)
                for i in range(n_ai):
                    flat = i * NUM_ENVS + j
                    last_masks[flat] = np.array(raw_masks[i], dtype=bool) if i < len(raw_masks) else np.ones(12, dtype=bool)

                if dones[j]:
                    episode_done[j] = True
                    next_env_states[j] = new_all_states[j]
                    for i in range(n_ai):
                        flat = i * NUM_ENVS + j
                        h[:, flat, :] = 0.0
                        c[:, flat, :] = 0.0
                        last_comm[flat] = 0.0
                        last_masks[flat] = True  # reset masks
                    env_kills[j] = infos[j].get("kill_count", 0)
                    env_wins[j] = 1 if infos[j].get("ai_win", False) else 0
                else:
                    next_env_states[j] = all_states[j]
            env_states = next_env_states

        # ── GAE ──
        all_advs = []
        for flat in range(FLAT_BATCH):
            rews = traj[flat]["rewards"]
            vals = traj[flat]["values"]
            if not rews:
                traj[flat]["advantages"] = []
                traj[flat]["returns"] = []
                continue
            advs = compute_gae(rews, vals)
            rets = [a + v for a, v in zip(advs, vals)]
            traj[flat]["advantages"] = advs
            traj[flat]["returns"] = rets
            all_advs.extend(advs)

        if len(all_advs) > 1:
            a_mean = float(np.mean(all_advs))
            a_std = float(np.std(all_advs)) + 1e-8
            for flat in range(FLAT_BATCH):
                traj[flat]["advantages"] = [
                    (a - a_mean) / a_std for a in traj[flat]["advantages"]]

        # ── 打包 batch tensor ──
        t_per_flat = [len(traj[f]["states"]) for f in range(FLAT_BATCH)]
        max_t = max(t_per_flat) if t_per_flat and max(t_per_flat) > 0 else 1

        bat_states = torch.zeros(max_t, FLAT_BATCH, NUM_CHANNELS, 15, 15)
        bat_scalars = torch.zeros(max_t, FLAT_BATCH, NUM_SCALARS)
        bat_actions = torch.zeros(max_t, FLAT_BATCH, 12)
        bat_old_lp = torch.zeros(max_t, FLAT_BATCH)
        bat_adv = torch.zeros(max_t, FLAT_BATCH)
        bat_ret = torch.zeros(max_t, FLAT_BATCH)
        bat_comm_acts = torch.zeros(max_t, FLAT_BATCH, NUM_COMM)
        bat_comm_lp = torch.zeros(max_t, FLAT_BATCH)
        bat_comm_mu = torch.zeros(max_t, FLAT_BATCH, NUM_COMM)
        bat_comm_logstd = torch.zeros(max_t, FLAT_BATCH, NUM_COMM)
        bat_masks = torch.ones(max_t, FLAT_BATCH, 12, dtype=torch.bool)
        mask = torch.zeros(max_t, FLAT_BATCH, dtype=torch.bool)

        for f in range(FLAT_BATCH):
            t_i = t_per_flat[f]
            if t_i == 0:
                continue
            mask[:t_i, f] = True
            bat_states[:t_i, f] = torch.stack([x[0] for x in traj[f]["states"]])
            bat_scalars[:t_i, f] = torch.stack([x[1] for x in traj[f]["states"]])
            bat_actions[:t_i, f] = torch.stack(traj[f]["actions"])
            bat_old_lp[:t_i, f] = torch.tensor(traj[f]["old_lp"])
            bat_adv[:t_i, f] = torch.tensor(traj[f]["advantages"])
            bat_ret[:t_i, f] = torch.tensor(traj[f]["returns"])
            if traj[f]["comm_acts"]:
                bat_comm_acts[:t_i, f] = torch.stack(traj[f]["comm_acts"])
                bat_comm_lp[:t_i, f] = torch.tensor(traj[f]["comm_lp"])
                bat_comm_mu[:t_i, f] = torch.stack(traj[f]["comm_mu"])
                bat_comm_logstd[:t_i, f] = torch.stack(traj[f]["comm_logstd"])
            if traj[f]["masks"]:
                bat_masks[:t_i, f] = torch.stack(traj[f]["masks"])

        bat_states = bat_states.to(device)
        bat_scalars = bat_scalars.to(device)
        bat_actions = bat_actions.to(device)
        bat_old_lp = bat_old_lp.to(device)
        bat_adv = bat_adv.to(device)
        bat_ret = bat_ret.to(device)
        bat_comm_acts = bat_comm_acts.to(device)
        bat_comm_lp = bat_comm_lp.to(device)
        bat_comm_mu = bat_comm_mu.to(device)
        bat_comm_logstd = bat_comm_logstd.to(device)
        bat_masks = bat_masks.to(device)
        mask = mask.to(device)

        # Fix 7: 預計算每個 flat index 的 team_id（用最後一次觀測到的 team）
        flat_team_ids = np.zeros(FLAT_BATCH, dtype=np.int32)
        for i in range(n_ai):
            for j in range(NUM_ENVS):
                flat = i * NUM_ENVS + j
                flat_team_ids[flat] = int(env_states[j][i][2]) if env_states[j][i] is not None else 0

        # ── PPO 更新 ──
        model.train()
        critic.train()
        for _ in range(PPO_EPOCHS):
            optimizer.zero_grad()
            optimizer_critic.zero_grad()
            h_tr = torch.zeros(1, FLAT_BATCH, HIDDEN_SIZE, device=device)
            c_tr = torch.zeros(1, FLAT_BATCH, HIDDEN_SIZE, device=device)
            ep_actor_losses = []
            ep_critic_losses = []

            for t in range(max_t):
                s = bat_states[t]
                sc = bat_scalars[t]
                a = bat_actions[t]
                lp_old = bat_old_lp[t]
                adv = bat_adv[t]
                ret = bat_ret[t]
                m_t = mask[t]
                old_ca = bat_comm_acts[t]
                old_clp = bat_comm_lp[t]
                m_act = bat_masks[t]   # (FLAT_BATCH, 12) action masks

                if not m_t.any():
                    with torch.no_grad():
                        _, _, _, _, (h_tr, c_tr) = model(s, sc, (h_tr, c_tr))
                    h_tr = h_tr.detach()
                    c_tr = c_tr.detach()
                    continue

                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits, new_mu, new_logstd, feat, (h_tr, c_tr) = model(
                        s, sc, (h_tr, c_tr))

                logits = logits.float()
                feat = feat.float()

                # Fix 2D: Action Masking 在 PPO update 時
                logits_masked = logits.masked_fill(~m_act, -1e9)
                probs = torch.sigmoid(logits_masked)
                dist_d = Bernoulli(probs)
                new_lp = dist_d.log_prob(a).sum(dim=1)
                entropy_d = dist_d.entropy().sum(dim=1)

                ratio = torch.exp(new_lp - lp_old)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv
                actor_loss = -torch.min(surr1, surr2)

                # 通訊損失
                new_std = torch.exp(new_logstd)
                new_comm_lp = CommHandler.old_log_prob(new_mu, new_logstd, old_ca)
                ratio_c = torch.exp(new_comm_lp - old_clp)
                surr1_c = ratio_c * adv
                surr2_c = torch.clamp(ratio_c, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv
                comm_loss = -torch.min(surr1_c, surr2_c)
                comm_entropy = Normal(new_mu, new_std + 1e-8).entropy().sum(dim=1)

                total_actor = (actor_loss + comm_loss
                               - entropy_coef * entropy_d
                               - COMM_ENT_COEF * comm_entropy)

                valid = m_t.float()
                n_valid = valid.sum().clamp(min=1)
                t_actor = (total_actor * valid).sum() / n_valid
                ep_actor_losses.append(t_actor)

                # Fix 1+7: Critic 使用 feat.detach()，按 team_id 分組
                feat_d = feat.detach()
                # 簡化分組：收集 team 0 和 team 1 的 feat
                team0_idx = [f for f in range(FLAT_BATCH) if m_t[f] and flat_team_ids[f] == 0]
                team1_idx = [f for f in range(FLAT_BATCH) if m_t[f] and flat_team_ids[f] == 1]

                if team0_idx:
                    t0_feat = feat_d[team0_idx]  # (N0, 256)
                else:
                    t0_feat = torch.zeros(1, HIDDEN_SIZE, device=device)
                if team1_idx:
                    t1_feat = feat_d[team1_idx]  # (N1, 256)
                else:
                    t1_feat = None

                # 計算 value — 所有 valid agent 共享同一個 critic output
                # 但每個 agent 各自用自己 team 的 pool
                v_pred_all = torch.zeros(FLAT_BATCH, device=device)
                if team0_idx:
                    v0 = critic(t0_feat, t1_feat)
                    for idx_i, fi in enumerate(team0_idx):
                        v_pred_all[fi] = v0[idx_i]
                if team1_idx:
                    # team1 的 critic：team1 在左，team0 在右
                    v1 = critic(t1_feat, t0_feat if team0_idx else None)
                    for idx_i, fi in enumerate(team1_idx):
                        v_pred_all[fi] = v1[idx_i]

                critic_l = (v_pred_all - ret).pow(2)
                t_critic = (critic_l * valid).sum() / n_valid
                ep_critic_losses.append(t_critic)

            # Actor backward
            if ep_actor_losses:
                total_actor_loss = torch.stack(ep_actor_losses).mean()
                scaler.scale(total_actor_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD)
                scaler.step(optimizer)

            # Critic backward
            if ep_critic_losses:
                total_critic_loss = torch.stack(ep_critic_losses).mean() * VALUE_COEF
                total_critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), MAX_GRAD)
                optimizer_critic.step()

            scaler.update()

        total_eps_done += NUM_ENVS
        batch_time = time.time() - batch_start_time
        elapsed = time.time() - start_time

        avg_rew_per_env = []
        for j in range(NUM_ENVS):
            env_total = 0.0
            for i in range(n_ai):
                flat = i * NUM_ENVS + j
                env_total += sum(traj[flat]["rewards"]) if traj[flat]["rewards"] else 0.0
            avg_rew_per_env.append(env_total / n_ai)
        avg_rew = float(np.mean(avg_rew_per_env))

        for k, w in zip(env_kills, env_wins):
            rolling_win.append((k, w))

        roll_list = list(rolling_win)
        total_ep = len(roll_list)
        dist_str = ""
        rolling_win_rate = 0.0

        if total_ep > 0:
            rolling_win_rate = sum(x[1] for x in roll_list) / total_ep
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
            save_checkpoint(path, model, critic, optimizer, optimizer_critic,
                            total_eps_done, current_stage, n_ai)
            logger.info(f"已儲存 checkpoint: {path}")
            save_milestone += SAVE_EVERY

    pbar.close()
    vec_env.close()

    final_path = f"snn_S{current_stage}_ep_final.pth"
    save_checkpoint(final_path, model, critic, optimizer, optimizer_critic,
                    total_eps_done, current_stage, n_ai)
    logger.info(f"訓練結束，已儲存: {final_path}")
    logger.info(f"中斷存檔指令: touch {STOP_SIGNAL_FILE}")


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="checkpoint path")
    parser.add_argument("--stage", type=int, default=None,
                        choices=[0, 1, 2, 3, 4, 5, 6], help="固定訓練階段")
    parser.add_argument("--stage_eps", type=int, default=50000, help="本階段訓練局數")
    parser.add_argument("--n_ai", type=int, default=2,
                        choices=[1, 2, 3, 4],
                        help="每個環境的 learning agent 數量")
    args = parser.parse_args()

    train(resume_path=args.resume, forced_stage=args.stage,
          target_stage_eps=args.stage_eps, n_ai=args.n_ai)

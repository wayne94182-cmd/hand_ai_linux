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

MANUAL_STOP = False


def _sigint_handler(_signum, _frame):
    global MANUAL_STOP
    MANUAL_STOP = True


signal.signal(signal.SIGINT, _sigint_handler)


def resolve_stage(resume_stage, forced_stage=None):
    if forced_stage is not None:
        return forced_stage
    return resume_stage


def compute_gae(rewards, values, last_value=0.0, truncated=False, gamma=GAMMA, lam=GAE_LAMBDA):
    advantages = []
    gae = 0.0
    # 若為 Timeout（非真實 done），使用 critic 估算的 last_value 作為 bootstrap
    next_v = last_value if truncated else 0.0
    for t in reversed(range(len(rewards))):
        if t < len(rewards) - 1:
            next_v_t = values[t + 1]
        else:
            next_v_t = next_v
        delta = rewards[t] + gamma * next_v_t - values[t]
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

def _unwrap(m):
    """取出 torch.compile wrapper 內的原始模型"""
    return getattr(m, "_orig_mod", m)


def save_checkpoint(path, model, critic, optimizer, optimizer_critic,
                    total_eps_done, stage_id, n_ai):
    payload = {
        "model_state": _unwrap(model).state_dict(),
        "critic_state": _unwrap(critic).state_dict(),
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

    # torch.compile 加速（僅 CUDA）
    torch.backends.cudnn.benchmark = False  # max_t 動態變化，不開 benchmark
    if device.type == "cuda":
        model  = torch.compile(model,  dynamic=True)
        critic = torch.compile(critic, dynamic=True)

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
        env_downs = [0] * NUM_ENVS
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
                    env_downs[j] = infos[j].get("down_count", 0)
                    env_wins[j] = 1 if infos[j].get("ai_win", False) else 0
                else:
                    next_env_states[j] = all_states[j]
            env_states = next_env_states

        # ── GAE ──
        # 判斷本次 rollout 是 done 還是 timeout（所有 env 都 done 表示真實 done）
        # 若有任何 env 是因達到 max_frames 而結束（timeout），需要 bootstrap
        # 此處簡化處理：每個 env 的結束情況已儲存在 episode_done 與 infos 裡，
        # 使用 info.get('timeout', False) 判斷；若無此欄位，預設為 False
        env_timeout = [infos[j].get("timeout", False) for j in range(NUM_ENVS)]

        # 對需要 bootstrap 的 flat index，用最後一步的 value 估算 last_value
        all_advs = []
        for flat in range(FLAT_BATCH):
            rews = traj[flat]["rewards"]
            vals = traj[flat]["values"]
            if not rews:
                traj[flat]["advantages"] = []
                traj[flat]["returns"] = []
                continue
            j = flat % NUM_ENVS
            is_timeout = env_timeout[j]
            # last_value: 若 timeout，使用最後一步的 critic value 作 bootstrap
            last_val = vals[-1] if is_timeout else 0.0
            advs = compute_gae(rews, vals, last_value=last_val, truncated=is_timeout)
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

        # pin_memory + non_blocking 加速 CPU→GPU 傳輸
        if device.type == "cuda":
            bat_states      = bat_states.pin_memory()
            bat_scalars     = bat_scalars.pin_memory()
            bat_actions     = bat_actions.pin_memory()
            bat_old_lp      = bat_old_lp.pin_memory()
            bat_adv         = bat_adv.pin_memory()
            bat_ret         = bat_ret.pin_memory()
            bat_comm_acts   = bat_comm_acts.pin_memory()
            bat_comm_lp     = bat_comm_lp.pin_memory()
            bat_comm_mu     = bat_comm_mu.pin_memory()
            bat_comm_logstd = bat_comm_logstd.pin_memory()
            bat_masks       = bat_masks.pin_memory()
            mask            = mask.pin_memory()

        bat_states      = bat_states.to(device, non_blocking=True)
        bat_scalars     = bat_scalars.to(device, non_blocking=True)
        bat_actions     = bat_actions.to(device, non_blocking=True)
        bat_old_lp      = bat_old_lp.to(device, non_blocking=True)
        bat_adv         = bat_adv.to(device, non_blocking=True)
        bat_ret         = bat_ret.to(device, non_blocking=True)
        bat_comm_acts   = bat_comm_acts.to(device, non_blocking=True)
        bat_comm_lp     = bat_comm_lp.to(device, non_blocking=True)
        bat_comm_mu     = bat_comm_mu.to(device, non_blocking=True)
        bat_comm_logstd = bat_comm_logstd.to(device, non_blocking=True)
        bat_masks       = bat_masks.to(device, non_blocking=True)
        mask            = mask.to(device, non_blocking=True)

        # Fix 7: 預計算每個 flat index 的 team_id（用最後一次觀測到的 team）
        flat_team_ids = np.zeros(FLAT_BATCH, dtype=np.int32)
        for i in range(n_ai):
            for j in range(NUM_ENVS):
                flat = i * NUM_ENVS + j
                flat_team_ids[flat] = int(env_states[j][i][2]) if env_states[j][i] is not None else 0

        # ── PPO 更新（seq_mode 向量化）──
        model.train()
        critic.train()

        # 預計算 team mask tensor（避免 PPO epoch 內重複建立）
        team0_mask_t = torch.tensor([flat_team_ids[f] == 0 for f in range(FLAT_BATCH)],
                                     dtype=torch.bool, device=device)
        team1_mask_t = torch.tensor([flat_team_ids[f] == 1 for f in range(FLAT_BATCH)],
                                     dtype=torch.bool, device=device)

        for _ in range(PPO_EPOCHS):
            optimizer.zero_grad()
            optimizer_critic.zero_grad()

            h_tr = torch.zeros(1, FLAT_BATCH, HIDDEN_SIZE, device=device)
            c_tr = torch.zeros(1, FLAT_BATCH, HIDDEN_SIZE, device=device)

            # 一次 forward 取代整個 for t 迴圈
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits_all, mu_all, logstd_all, feat_all, _ = model(
                    bat_states, bat_scalars, (h_tr, c_tr),
                    comm_in=None,
                    seq_mode=True
                )
            # logits_all: (max_t, FLAT_BATCH, 12)
            # feat_all:   (max_t, FLAT_BATCH, 256)

            logits_all = logits_all.float()
            feat_all   = feat_all.float()

            # 展平所有時間步 → (max_t * FLAT_BATCH, ...)
            TB = max_t * FLAT_BATCH
            valid_flat = mask.reshape(TB)                             # (TB,)
            n_valid = valid_flat.float().sum().clamp(min=1)

            logits_flat = logits_all.reshape(TB, 12)
            mu_flat     = mu_all.reshape(TB, NUM_COMM)
            logstd_flat = logstd_all.reshape(TB, NUM_COMM)
            feat_flat   = feat_all.reshape(TB, HIDDEN_SIZE)

            a_flat      = bat_actions.reshape(TB, 12)
            lp_old_flat = bat_old_lp.reshape(TB)
            adv_flat    = bat_adv.reshape(TB)
            ret_flat    = bat_ret.reshape(TB)
            ca_flat     = bat_comm_acts.reshape(TB, NUM_COMM)
            clp_flat    = bat_comm_lp.reshape(TB)
            m_act_flat  = bat_masks.reshape(TB, 12)

            # Action Masking
            logits_masked = logits_flat.masked_fill(~m_act_flat, -1e9)
            probs = torch.sigmoid(logits_masked)
            dist_d = Bernoulli(probs)
            new_lp = dist_d.log_prob(a_flat).sum(dim=1)         # (TB,)
            entropy_d = dist_d.entropy().sum(dim=1)             # (TB,)

            ratio = torch.exp(new_lp - lp_old_flat)
            surr1 = ratio * adv_flat
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_flat
            actor_loss = -torch.min(surr1, surr2)

            # 通訊損失
            new_std = torch.exp(logstd_flat)
            new_comm_lp = CommHandler.old_log_prob(mu_flat, logstd_flat, ca_flat)
            ratio_c = torch.exp(new_comm_lp - clp_flat)
            surr1_c = ratio_c * adv_flat
            surr2_c = torch.clamp(ratio_c, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_flat
            comm_loss = -torch.min(surr1_c, surr2_c)
            comm_entropy = Normal(mu_flat, new_std + 1e-8).entropy().sum(dim=1)

            total_actor = (actor_loss + comm_loss
                           - entropy_coef * entropy_d
                           - COMM_ENT_COEF * comm_entropy)

            t_actor_loss = (total_actor * valid_flat.float()).sum() / n_valid

            # Critic（需按 team 分組）
            feat_d = feat_flat.detach()   # (TB, 256)

            # 展平 team mask: (max_t, FLAT_BATCH) → (TB,)
            # 每個時間步的 team 分配都一樣，所以直接 expand
            team0_exp = team0_mask_t.unsqueeze(0).expand(max_t, -1).reshape(TB)
            team1_exp = team1_mask_t.unsqueeze(0).expand(max_t, -1).reshape(TB)

            # valid + team
            v0_idx = valid_flat & team0_exp
            v1_idx = valid_flat & team1_exp

            v_pred_flat = torch.zeros(TB, device=device)

            if v0_idx.any():
                t0_feat = feat_d[v0_idx]
                t1_feat = feat_d[v1_idx] if v1_idx.any() else None
                v0 = critic(t0_feat, t1_feat)
                v_pred_flat[v0_idx] = v0

            if v1_idx.any():
                t1_feat = feat_d[v1_idx]
                t0_feat = feat_d[v0_idx] if v0_idx.any() else None
                v1 = critic(t1_feat, t0_feat)
                v_pred_flat[v1_idx] = v1

            critic_l = (v_pred_flat - ret_flat).pow(2)
            t_critic_loss = (critic_l * valid_flat.float()).sum() / n_valid

            # Actor backward
            scaler.scale(t_actor_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(_unwrap(model).parameters(), MAX_GRAD)
            scaler.step(optimizer)

            # Critic backward
            (t_critic_loss * VALUE_COEF).backward()
            torch.nn.utils.clip_grad_norm_(_unwrap(critic).parameters(), MAX_GRAD)
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

        for d, w in zip(env_downs, env_wins):
            rolling_win.append((d, w))

        roll_list = list(rolling_win)
        total_ep = len(roll_list)
        dist_str = ""
        rolling_win_rate = 0.0

        if total_ep > 0:
            rolling_win_rate = sum(x[1] for x in roll_list) / total_ep
            if stage_spec.mode == "scripted":
                max_e = stage_spec.enemy_count
                downs_only = [x[0] for x in roll_list]
                dist = [downs_only.count(i) / total_ep for i in range(max_e + 1)]
                dist_str = " ".join([f"d{i}:{p:.2f}" for i, p in enumerate(dist)])

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
                postfix["downs"] = dist_str
        pbar.set_postfix(postfix)

        msg = (
            f"進度: Stage{current_stage}-{stage_spec.name}, eps={total_eps_done}, "
            f"avg_rew={avg_rew:.2f}, ent={entropy_coef:.3f}"
        )
        if show_stats:
            msg += f", win_rate={rolling_win_rate:.3f}"
            if dist_str:
                msg += f", downs=[{dist_str}]"
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

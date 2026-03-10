"""
train.py — MAPPO + N-Agent Flatten 訓練腳本
支援多個 learning agents、LSTM 隱藏狀態管理、
通訊向量、action masking、以及分離的 Actor / Critic 優化。
修正：feat.detach / action masking / comm team filter / critic team grouping
優化：共享內存零拷貝、預分配緩衝區、Numba 加速
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

from ai import ConvLSTM, TeamPoolingCritic, CommHandler, HIDDEN_SIZE, NUM_COMM, NUM_ACTIONS_DISCRETE
from game import GameEnv, get_stage_spec
from game.env import NUM_CHANNELS, NUM_SCALARS
from game.config import VIEW_SIZE, ROWS, COLS

# GPU 渲染器（僅在非 render_mode 時使用）
from gpu_renderer import GPURenderer, pack_raw_states_to_tensors

# Padding 上限常數
MAX_ALLIES = 3
MAX_ENEMIES = 8
MAX_ITEMS = 20
MAX_THREATS = 32
MAX_SOUNDS = 16

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
NUM_ENVS = 96
ROLLOUT_STEPS = 512  # 固定步數採樣（Fixed-Length Rollout）
GAMMA = 0.990
GAE_LAMBDA = 0.95
PPO_EPOCHS = 4
CLIP_EPS = 0.2
VALUE_COEF = 0.5
ENT_START = 0.05
ENT_END = 0.02
LR = 1.5e-4
MAX_GRAD = 0.5

ROLLING_WIN_WINDOW = 200
SAVE_EVERY = 5000
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
        _unwrap(model).load_state_dict(ckpt["model_state"])
        if optimizer is not None and "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if critic is not None and "critic_state" in ckpt:
            _unwrap(critic).load_state_dict(ckpt["critic_state"])
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

def train(resume_path=None, forced_stage=None, target_stage_eps=50000, n_ai=2, use_gpu_renderer=False):
    """
    新增參數：
        use_gpu_renderer: 是否使用 GPU 端即時渲染（實驗性功能）
    """
    global MANUAL_STOP

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"訓練裝置: {device}, N_AI={n_ai}, GPU_Renderer={use_gpu_renderer}")

    FLAT_BATCH = n_ai * NUM_ENVS

    model = ConvLSTM().to(device)
    critic = TeamPoolingCritic().to(device)

    # GPU 優化：使用 foreach=True（批量更新參數，兼容 AMP）
    # 注意：fused=True 不能與 GradScaler 一起使用，所以用 foreach
    use_foreach = device.type == "cuda"
    optimizer = optim.AdamW(model.parameters(), lr=LR, foreach=use_foreach)
    optimizer_critic = optim.AdamW(critic.parameters(), lr=LR, foreach=use_foreach)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # torch.compile 加速（僅 CUDA）
    torch.backends.cudnn.benchmark = True   # 固定形狀，啟用 cuDNN 自動優化
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
                                        n_learning_agents=nai,
                                        use_gpu_renderer=use_gpu_renderer),
        num_envs=NUM_ENVS,
        stage_id=current_stage,
        n_ai=n_ai,
    )

    batch_episodes = max(1, target_stage_eps // NUM_ENVS)
    start_batch = total_eps_done // NUM_ENVS
    start_time = time.time()

    rolling_win = deque(maxlen=ROLLING_WIN_WINDOW)

    # ═══════════════════════════════════════════════════════
    # 訓練開始前：初始化環境與 LSTM 狀態（只做一次）
    # ═══════════════════════════════════════════════════════
    vec_env.set_stage(current_stage)

    # GPU 渲染器初始化（僅在啟用時）
    gpu_renderer = None
    if use_gpu_renderer:
        # 假設所有地圖大小相同（從 config 讀取）
        from game.config import ROWS, COLS
        gpu_renderer = GPURenderer(map_rows=ROWS, map_cols=COLS)
        logger.info(f"GPU 渲染器已啟用：map_size=({ROWS}, {COLS})")

    # LSTM 隱藏狀態：跨 batch 持續（只在 episode done 時清空）
    h = torch.zeros(1, FLAT_BATCH, HIDDEN_SIZE, device=device)
    c = torch.zeros(1, FLAT_BATCH, HIDDEN_SIZE, device=device)
    last_comm = np.zeros((FLAT_BATCH, NUM_COMM), dtype=np.float32)

    # 環境狀態：跨 batch 持續（只在 episode done 時重置）
    env_states = vec_env.reset()

    # Action mask：跨 batch 持續
    last_masks = np.ones((FLAT_BATCH, NUM_ACTIONS_DISCRETE), dtype=bool)

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

        stage_spec = get_stage_spec(current_stage)

        progress = total_eps_done / max(1, target_stage_eps)
        entropy_coef = ENT_START + (ENT_END - ENT_START) * progress

        model.eval()

        # 固定步數採樣：預分配緩衝區
        # 修正：GPU 渲染模式下使用 NumPy Array 存座標，徹底消除 GC 災難！
        if use_gpu_renderer:
            # 預先分配 NumPy 緩衝區存儲座標（避免 Dictionary 碎片化）
            buf_agent_poses = np.zeros((ROLLOUT_STEPS, FLAT_BATCH, 4), dtype=np.float32)  # [x, y, angle, is_sniper]
            buf_ally_poses = np.zeros((ROLLOUT_STEPS, FLAT_BATCH, MAX_ALLIES, 3), dtype=np.float32)
            buf_ally_mask = np.zeros((ROLLOUT_STEPS, FLAT_BATCH, MAX_ALLIES), dtype=bool)
            buf_enemy_poses = np.zeros((ROLLOUT_STEPS, FLAT_BATCH, MAX_ENEMIES, 3), dtype=np.float32)
            buf_enemy_mask = np.zeros((ROLLOUT_STEPS, FLAT_BATCH, MAX_ENEMIES), dtype=bool)
            buf_item_poses = np.zeros((ROLLOUT_STEPS, FLAT_BATCH, MAX_ITEMS, 3), dtype=np.float32)
            buf_item_mask = np.zeros((ROLLOUT_STEPS, FLAT_BATCH, MAX_ITEMS), dtype=bool)
            buf_threat_poses = np.zeros((ROLLOUT_STEPS, FLAT_BATCH, MAX_THREATS, 3), dtype=np.float32)
            buf_threat_mask = np.zeros((ROLLOUT_STEPS, FLAT_BATCH, MAX_THREATS), dtype=bool)
            buf_sound_waves = np.zeros((ROLLOUT_STEPS, FLAT_BATCH, MAX_SOUNDS, 4), dtype=np.float32)
            buf_sound_mask = np.zeros((ROLLOUT_STEPS, FLAT_BATCH, MAX_SOUNDS), dtype=bool)
            buf_grids = np.zeros((ROLLOUT_STEPS, FLAT_BATCH, ROWS, COLS), dtype=np.float32)
            buf_poison_info = np.zeros((ROLLOUT_STEPS, FLAT_BATCH, 4), dtype=np.float32)

            # GPU 模式也需要緩存圖片！避免 PPO 更新時重新渲染（雙重渲染災難）
            # 使用 Tensor 存在 GPU 上，避免 CPU↔GPU 搬運
            buf_states_gpu = torch.zeros((ROLLOUT_STEPS, FLAT_BATCH, NUM_CHANNELS, VIEW_SIZE, VIEW_SIZE),
                                         dtype=torch.float32, device=device)
        else:
            # CPU 渲染模式：預存圖片
            buf_states = np.zeros((ROLLOUT_STEPS, FLAT_BATCH, NUM_CHANNELS, VIEW_SIZE, VIEW_SIZE), dtype=np.float32)
        buf_scalars = np.zeros((ROLLOUT_STEPS, FLAT_BATCH, NUM_SCALARS), dtype=np.float32)
        buf_actions = np.zeros((ROLLOUT_STEPS, FLAT_BATCH, NUM_ACTIONS_DISCRETE), dtype=np.float32)
        buf_values = np.zeros((ROLLOUT_STEPS, FLAT_BATCH), dtype=np.float32)
        buf_old_lp = np.zeros((ROLLOUT_STEPS, FLAT_BATCH), dtype=np.float32)
        buf_comm_acts = np.zeros((ROLLOUT_STEPS, FLAT_BATCH, NUM_COMM), dtype=np.float32)
        buf_comm_lp = np.zeros((ROLLOUT_STEPS, FLAT_BATCH), dtype=np.float32)
        buf_comm_mu = np.zeros((ROLLOUT_STEPS, FLAT_BATCH, NUM_COMM), dtype=np.float32)
        buf_comm_logstd = np.zeros((ROLLOUT_STEPS, FLAT_BATCH, NUM_COMM), dtype=np.float32)
        buf_masks = np.ones((ROLLOUT_STEPS, FLAT_BATCH, NUM_ACTIONS_DISCRETE), dtype=bool)
        buf_rewards = np.zeros((ROLLOUT_STEPS, FLAT_BATCH), dtype=np.float32)
        buf_dones = np.zeros((ROLLOUT_STEPS, FLAT_BATCH), dtype=bool)  # 追蹤 episode 邊界

        # 統計信息
        total_episode_count = 0
        # 記錄本次 rollout 中完成的所有 episode 的統計 (down_count, is_win)
        completed_episodes = []

        # 固定步數採樣循環
        for step in range(ROLLOUT_STEPS):
            # ── 分支 1：GPU 渲染模式（實驗性） ──
            if use_gpu_renderer:
                # 1. 從 env_states 收集 raw_state（Dict 格式）並存入 NumPy Arrays
                sc_np = np.zeros((NUM_ENVS, n_ai, NUM_SCALARS), dtype=np.float32)
                team_ids = np.zeros((NUM_ENVS, n_ai), dtype=np.int32)

                for j in range(NUM_ENVS):
                    for i in range(n_ai):
                        rs = env_states[j][i]  # Dict: raw_state
                        flat = i * NUM_ENVS + j

                        # 存入預配置 NumPy 緩衝區（取代 Dictionary append）
                        buf_agent_poses[step, flat] = rs["agent_pose"]

                        # Ally poses (padding)
                        ally_list = rs["ally_poses"]
                        n_allies = min(len(ally_list), MAX_ALLIES)
                        buf_ally_mask[step, flat, :] = False
                        if n_allies > 0:
                            buf_ally_poses[step, flat, :n_allies] = ally_list[:n_allies]
                            buf_ally_mask[step, flat, :n_allies] = True

                        # Enemy poses (padding)
                        enemy_list = rs["enemy_poses"]
                        n_enemies = min(len(enemy_list), MAX_ENEMIES)
                        buf_enemy_mask[step, flat, :] = False
                        if n_enemies > 0:
                            buf_enemy_poses[step, flat, :n_enemies] = enemy_list[:n_enemies]
                            buf_enemy_mask[step, flat, :n_enemies] = True

                        # Item poses (padding)
                        item_list = rs["item_poses"]
                        n_items = min(len(item_list), MAX_ITEMS)
                        buf_item_mask[step, flat, :] = False
                        if n_items > 0:
                            buf_item_poses[step, flat, :n_items] = item_list[:n_items]
                            buf_item_mask[step, flat, :n_items] = True

                        # Threat poses (padding)
                        threat_list = rs["threat_poses"]
                        n_threats = min(len(threat_list), MAX_THREATS)
                        buf_threat_mask[step, flat, :] = False
                        if n_threats > 0:
                            buf_threat_poses[step, flat, :n_threats] = threat_list[:n_threats]
                            buf_threat_mask[step, flat, :n_threats] = True

                        # Sound waves (padding)
                        sound_list = rs["sound_waves"]
                        n_sounds = min(len(sound_list), MAX_SOUNDS)
                        buf_sound_mask[step, flat, :] = False
                        if n_sounds > 0:
                            buf_sound_waves[step, flat, :n_sounds] = sound_list[:n_sounds]
                            buf_sound_mask[step, flat, :n_sounds] = True

                        # Grid and poison info
                        buf_grids[step, flat] = rs["grid"]
                        buf_poison_info[step, flat] = rs["poison_info"]

                        # Scalars and team_id
                        sc_np[j, i] = rs["scalars"]
                        team_ids[j, i] = int(rs["team_id"])

                # 2. 即時渲染當前步（用於推理決策）
                # 優化：使用 non_blocking 傳輸加速
                with torch.no_grad():
                    agent_poses_t = torch.from_numpy(buf_agent_poses[step]).to(device, non_blocking=True)
                    ally_poses_t = torch.from_numpy(buf_ally_poses[step]).to(device, non_blocking=True)
                    ally_mask_t = torch.from_numpy(buf_ally_mask[step]).to(device, non_blocking=True)
                    enemy_poses_t = torch.from_numpy(buf_enemy_poses[step]).to(device, non_blocking=True)
                    enemy_mask_t = torch.from_numpy(buf_enemy_mask[step]).to(device, non_blocking=True)
                    item_poses_t = torch.from_numpy(buf_item_poses[step]).to(device, non_blocking=True)
                    item_mask_t = torch.from_numpy(buf_item_mask[step]).to(device, non_blocking=True)
                    threat_poses_t = torch.from_numpy(buf_threat_poses[step]).to(device, non_blocking=True)
                    threat_mask_t = torch.from_numpy(buf_threat_mask[step]).to(device, non_blocking=True)
                    sound_waves_t = torch.from_numpy(buf_sound_waves[step]).to(device, non_blocking=True)
                    sound_mask_t = torch.from_numpy(buf_sound_mask[step]).to(device, non_blocking=True)
                    grids_t = torch.from_numpy(buf_grids[step]).to(device, non_blocking=True)
                    poison_info_t = torch.from_numpy(buf_poison_info[step]).to(device, non_blocking=True)

                    # GPU 渲染並緩存
                    s_t = gpu_renderer.render_batch(
                        agent_poses_t, ally_poses_t, ally_mask_t,
                        enemy_poses_t, enemy_mask_t,
                        item_poses_t, item_mask_t,
                        threat_poses_t, threat_mask_t,
                        sound_waves_t, sound_mask_t,
                        grids_t, poison_info_t, device=device
                    )

                    # 存入 GPU 緩存（PPO 更新時使用）
                    buf_states_gpu[step] = s_t

                # 3. Scalars
                sc_flat = sc_np.transpose(1, 0, 2).reshape(FLAT_BATCH, NUM_SCALARS)
                sc_t = torch.as_tensor(sc_flat, dtype=torch.float32, device=device)

            # ── 分支 2：原始 CPU 渲染模式（向後相容） ──
            else:
                # 1. 從 NUM_ENVS 個環境收集狀態
                s_np = np.zeros((NUM_ENVS, n_ai, NUM_CHANNELS, VIEW_SIZE, VIEW_SIZE), dtype=np.float32)
                sc_np = np.zeros((NUM_ENVS, n_ai, NUM_SCALARS), dtype=np.float32)
                team_ids = np.zeros((NUM_ENVS, n_ai), dtype=np.int32)
                for j in range(NUM_ENVS):
                    for i in range(n_ai):
                        s_np[j, i] = env_states[j][i][0]
                        sc_np[j, i] = env_states[j][i][1]
                        team_ids[j, i] = int(env_states[j][i][2])

                # 2. 展平：先 AI index，再 env index
                s_flat = s_np.transpose(1, 0, 2, 3, 4).reshape(FLAT_BATCH, NUM_CHANNELS, VIEW_SIZE, VIEW_SIZE)
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

            # 存軌跡（修正：GPU 渲染模式不存圖片）
            for i in range(n_ai):
                for j in range(NUM_ENVS):
                    flat = i * NUM_ENVS + j
                    if not use_gpu_renderer:
                        buf_states[step, flat] = s_flat[flat]  # 只有 CPU 模式需要存圖片
                    buf_scalars[step, flat] = sc_flat[flat]
                    buf_actions[step, flat] = acts_np[flat]
                    buf_values[step, flat] = vals_np[flat]
                    buf_old_lp[step, flat] = lp_disc_np[flat]
                    buf_comm_acts[step, flat] = comm_np_new[flat]
                    buf_comm_lp[step, flat] = lp_comm_np[flat]
                    buf_comm_mu[step, flat] = mu_np[flat]
                    buf_comm_logstd[step, flat] = logstd_np[flat]
                    buf_masks[step, flat] = last_masks[flat]

            # 6. 還原形狀派發給 env
            acts_env = acts_np.reshape(n_ai, NUM_ENVS, NUM_ACTIONS_DISCRETE).transpose(1, 0, 2)
            last_comm = comm_np_new.copy()

            env_actions = []
            for j in range(NUM_ENVS):
                env_actions.append([acts_env[j][i].tolist() for i in range(n_ai)])

            vec_env.step_async(env_actions, FRAME_SKIP)
            all_states, all_rewards, dones, new_all_states, infos = vec_env.step_wait()

            # 處理 reward 和環境重置
            next_env_states = list(env_states)
            for j in range(NUM_ENVS):
                # 存 reward
                for i in range(n_ai):
                    flat = i * NUM_ENVS + j
                    rew = all_rewards[j][i] if isinstance(all_rewards[j], list) else all_rewards[j]
                    buf_rewards[step, flat] = rew
                    buf_dones[step, flat] = dones[j]  # 記錄 episode 邊界

                # 更新 action mask
                raw_masks = infos[j].get("action_masks", [[True]*NUM_ACTIONS_DISCRETE for _ in range(n_ai)])
                for i in range(n_ai):
                    flat = i * NUM_ENVS + j
                    last_masks[flat] = np.array(raw_masks[i], dtype=bool) if i < len(raw_masks) else np.ones(NUM_ACTIONS_DISCRETE, dtype=bool)

                # 環境 done 時立刻 reset（無縫接軌）
                if dones[j]:
                    # 統計信息：記錄完成的 episode
                    total_episode_count += 1
                    down_count = infos[j].get("down_count", 0)
                    is_win = 1 if infos[j].get("ai_win", False) else 0
                    completed_episodes.append((down_count, is_win))

                    # 立刻重置環境
                    next_env_states[j] = new_all_states[j]

                    # 清空 LSTM 狀態（防止記憶污染）
                    for i in range(n_ai):
                        flat = i * NUM_ENVS + j
                        h[:, flat, :] = 0.0
                        c[:, flat, :] = 0.0
                        last_comm[flat] = 0.0
                        last_masks[flat] = True
                else:
                    next_env_states[j] = all_states[j]

            env_states = next_env_states

        # ── GAE（固定步數，處理跨 episode 邊界）──
        buf_advantages = np.zeros((ROLLOUT_STEPS, FLAT_BATCH), dtype=np.float32)
        buf_returns = np.zeros((ROLLOUT_STEPS, FLAT_BATCH), dtype=np.float32)

        # 對每個 flat index 計算 GAE（處理 done 標記作為 episode 邊界）
        all_advs = []
        for flat in range(FLAT_BATCH):
            rews = buf_rewards[:, flat]
            vals = buf_values[:, flat]
            ep_dones = buf_dones[:, flat]

            # 分段計算 GAE（根據 done 標記切分 episode）
            episode_start = 0
            for t in range(ROLLOUT_STEPS):
                if ep_dones[t] or t == ROLLOUT_STEPS - 1:
                    # Episode 結束，計算這段的 GAE
                    episode_end = t + 1
                    ep_rews = rews[episode_start:episode_end].tolist()
                    ep_vals = vals[episode_start:episode_end].tolist()

                    # Bootstrap: done 時 last_value=0，否則使用最後的 value
                    last_val = 0.0 if ep_dones[t] else ep_vals[-1]
                    ep_advs = compute_gae(ep_rews, ep_vals, last_value=last_val, truncated=not ep_dones[t])
                    ep_rets = [a + v for a, v in zip(ep_advs, ep_vals)]

                    buf_advantages[episode_start:episode_end, flat] = ep_advs
                    buf_returns[episode_start:episode_end, flat] = ep_rets
                    all_advs.extend(ep_advs)

                    episode_start = episode_end

        # Normalize advantages
        if len(all_advs) > 1:
            a_mean = float(np.mean(all_advs))
            a_std = float(np.std(all_advs)) + 1e-8
            buf_advantages = (buf_advantages - a_mean) / a_std

        # ── 打包 batch tensor──
        if use_gpu_renderer:
            # GPU 模式：直接使用 Rollout 階段緩存的圖片（已在 GPU 上，零拷貝）
            bat_states = buf_states_gpu  # (ROLLOUT_STEPS, FLAT_BATCH, 10, 15, 15)
        else:
            # CPU 渲染模式：從 Buffer 讀取並傳輸到 GPU
            bat_states = torch.as_tensor(buf_states, dtype=torch.float32).to(device)

        # 其他 Tensor（零拷貝 + pin_memory 優化）
        bat_scalars = torch.as_tensor(buf_scalars, dtype=torch.float32)
        bat_actions = torch.as_tensor(buf_actions, dtype=torch.float32)
        bat_old_lp = torch.as_tensor(buf_old_lp, dtype=torch.float32)
        bat_adv = torch.as_tensor(buf_advantages, dtype=torch.float32)
        bat_ret = torch.as_tensor(buf_returns, dtype=torch.float32)
        bat_comm_acts = torch.as_tensor(buf_comm_acts, dtype=torch.float32)
        bat_comm_lp = torch.as_tensor(buf_comm_lp, dtype=torch.float32)
        bat_comm_mu = torch.as_tensor(buf_comm_mu, dtype=torch.float32)
        bat_comm_logstd = torch.as_tensor(buf_comm_logstd, dtype=torch.float32)
        bat_masks = torch.as_tensor(buf_masks, dtype=torch.bool)

        # pin_memory + non_blocking 加速 CPU→GPU 傳輸
        if device.type == "cuda":
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

        # Fix 7: 預計算每個 flat index 的 team_id（用最後一次觀測到的 team）
        # 修正：相容 Tuple（CPU）與 Dict（GPU）兩種資料結構
        flat_team_ids = np.zeros(FLAT_BATCH, dtype=np.int32)
        for i in range(n_ai):
            for j in range(NUM_ENVS):
                flat = i * NUM_ENVS + j
                state = env_states[j][i]
                if state is None:
                    flat_team_ids[flat] = 0
                elif isinstance(state, dict):
                    # GPU 渲染模式：Dictionary 格式
                    flat_team_ids[flat] = int(state["team_id"])
                else:
                    # CPU 渲染模式：Tuple 格式 (view, scalars, team_id)
                    flat_team_ids[flat] = int(state[2])

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
            # logits_all: (ROLLOUT_STEPS, FLAT_BATCH, NUM_ACTIONS_DISCRETE)
            # feat_all:   (ROLLOUT_STEPS, FLAT_BATCH, HIDDEN_SIZE)

            logits_all = logits_all.float()
            feat_all   = feat_all.float()

            # 展平所有時間步 → (ROLLOUT_STEPS * FLAT_BATCH, ...)
            TB = ROLLOUT_STEPS * FLAT_BATCH

            logits_flat = logits_all.reshape(TB, NUM_ACTIONS_DISCRETE)
            mu_flat     = mu_all.reshape(TB, NUM_COMM)
            logstd_flat = logstd_all.reshape(TB, NUM_COMM)
            feat_flat   = feat_all.reshape(TB, HIDDEN_SIZE)

            a_flat      = bat_actions.reshape(TB, NUM_ACTIONS_DISCRETE)
            lp_old_flat = bat_old_lp.reshape(TB)
            adv_flat    = bat_adv.reshape(TB)
            ret_flat    = bat_ret.reshape(TB)
            ca_flat     = bat_comm_acts.reshape(TB, NUM_COMM)
            clp_flat    = bat_comm_lp.reshape(TB)
            m_act_flat  = bat_masks.reshape(TB, NUM_ACTIONS_DISCRETE)

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

            t_actor_loss = total_actor.mean()

            # Critic（需按 team 分組，還原為 B_env 維度進行 Team Pooling）
            # 移除 detach()，讓 Critic 梯度能回傳到 CNN（修復 Task 3）
            # feat_all 的形狀為 (ROLLOUT_STEPS, FLAT_BATCH, HIDDEN_SIZE)
            # FLAT_BATCH = n_ai * NUM_ENVS
            feat_env = feat_all.view(ROLLOUT_STEPS, n_ai, NUM_ENVS, HIDDEN_SIZE).transpose(1, 2).reshape(ROLLOUT_STEPS * NUM_ENVS, n_ai, HIDDEN_SIZE)
            
            # 從 precomputed flat_team_ids 推導每個 AI 的 team (以 j=0 也就是第一個 env 為代表)
            team0_mask_ai = torch.tensor([flat_team_ids[i * NUM_ENVS] == 0 for i in range(n_ai)], dtype=torch.bool, device=device)
            team1_mask_ai = torch.tensor([flat_team_ids[i * NUM_ENVS] == 1 for i in range(n_ai)], dtype=torch.bool, device=device)

            v_pred_env = torch.zeros(ROLLOUT_STEPS * NUM_ENVS, n_ai, device=device)

            if team0_mask_ai.any():
                t0_feat = feat_env[:, team0_mask_ai, :]  # (B_env, N0, 256)
                t1_feat = feat_env[:, team1_mask_ai, :] if team1_mask_ai.any() else None
                v0_env = critic(t0_feat, t1_feat)  # (B_env,)
                # 廣播給同隊所有 agent
                v_pred_env[:, team0_mask_ai] = v0_env.unsqueeze(1)

            if team1_mask_ai.any():
                t1_feat = feat_env[:, team1_mask_ai, :]
                t0_feat = feat_env[:, team0_mask_ai, :] if team0_mask_ai.any() else None
                v1_env = critic(t1_feat, t0_feat)
                v_pred_env[:, team1_mask_ai] = v1_env.unsqueeze(1)

            # 將 v_pred_env 變換回 (TB,)
            # (ROLLOUT_STEPS * NUM_ENVS, n_ai) -> (ROLLOUT_STEPS, NUM_ENVS, n_ai)
            # transpose(1, 2) -> (ROLLOUT_STEPS, n_ai, NUM_ENVS) -> reshape(TB)
            v_pred_flat = v_pred_env.view(ROLLOUT_STEPS, NUM_ENVS, n_ai).transpose(1, 2).reshape(TB)

            critic_l = (v_pred_flat - ret_flat).pow(2)
            t_critic_loss = critic_l.mean()

            # Task 3 修復：合併 Actor + Critic loss 一次 backward，讓梯度能回傳到共享 CNN
            # 不能分別 backward，否則第二次會出現 "backward through the graph a second time" 錯誤
            total_loss = t_actor_loss + t_critic_loss * VALUE_COEF
            scaler.scale(total_loss).backward()

            scaler.unscale_(optimizer)
            scaler.unscale_(optimizer_critic)
            torch.nn.utils.clip_grad_norm_(_unwrap(model).parameters(), MAX_GRAD)
            torch.nn.utils.clip_grad_norm_(_unwrap(critic).parameters(), MAX_GRAD)

            scaler.step(optimizer)
            scaler.step(optimizer_critic)
            scaler.update()

        total_eps_done += NUM_ENVS
        batch_time = time.time() - batch_start_time
        elapsed = time.time() - start_time

        # 固定步數採樣：計算 rollout 內的平均獎勵
        # 注意：這是 rollout 片段的獎勵，不是完整 episode
        avg_rew_per_env = []
        for j in range(NUM_ENVS):
            env_total = 0.0
            for i in range(n_ai):
                flat = i * NUM_ENVS + j
                # 所有 flat index 都收集了完整的 ROLLOUT_STEPS
                env_total += buf_rewards[:, flat].sum()
            avg_rew_per_env.append(env_total / n_ai)
        avg_rew = float(np.mean(avg_rew_per_env))

        # 將本次 rollout 中完成的所有 episode 加入滾動統計
        for d, w in completed_episodes:
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
    parser.add_argument("--use_gpu_renderer", action="store_true",
                        help="啟用 GPU 端即時渲染（實驗性功能）")
    args = parser.parse_args()

    train(resume_path=args.resume, forced_stage=args.stage,
          target_stage_eps=args.stage_eps, n_ai=args.n_ai,
          use_gpu_renderer=args.use_gpu_renderer)

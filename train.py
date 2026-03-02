import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Bernoulli
import numpy as np
from tqdm import tqdm
import logging
import sys
import argparse
import multiprocessing as mp
import time

from ai import ConvSNN, HIDDEN_SIZE

# ==========================================
# 記錄日誌設置
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("train.log", mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ==========================================
# 訓練超參數
# ==========================================
FRAME_SKIP   = 2
NUM_ENVS     = 96
GAMMA        = 0.998
GAE_LAMBDA   = 0.95
PPO_EPOCHS   = 4
CLIP_EPS     = 0.2
VALUE_COEF   = 0.5
ENT_START    = 0.10
ENT_END      = 0.01
LR           = 3e-4
MAX_GRAD     = 0.5
TOTAL_EPS    = 50000
TBPTT_LEN    = 20


# ==========================================
# SubprocVecEnv：Worker 函式（子進程中執行）
# ==========================================
def _worker(remote, parent_remote, env_fn):
    """
    每個子進程持有一個獨立的 GameEnv 實例。
    透過 Pipe 接收主進程指令，執行後回傳結果。
    若 env.step 後 done==True，Worker 自動 reset 並回傳新 state。
    """
    # 子進程不需要父端的 pipe
    parent_remote.close()
    env = env_fn()

    try:
        while True:
            cmd, data = remote.recv()

            if cmd == 'step':
                act1, act2, frame_skip = data
                states, rewards, done, info = env.step(act1, act2, frame_skip=frame_skip)
                # ★ done 後自動 reset，但把 done 旗標傳回讓主進程能歸零 hidden state
                if done:
                    new_states = env.reset()
                    remote.send((states, rewards, True, new_states))
                else:
                    remote.send((states, rewards, False, None))

            elif cmd == 'reset':
                states = env.reset()
                remote.send(states)

            elif cmd == 'set_config':
                env.set_config(*data)

            elif cmd == 'close':
                remote.close()
                break

            else:
                raise NotImplementedError(f"Unknown command: {cmd}")

    except KeyboardInterrupt:
        pass
    finally:
        env_done = getattr(env, 'render_mode', False)
        if env_done:
            import pygame
            pygame.quit()


# ==========================================
# VecEnv：主進程側的管理器
# ==========================================
class VecEnv:
    """
    管理 num_envs 個子進程，每個子進程持有一個 GameEnv。
    使用 step_async / step_wait 分離「發送動作」與「收集結果」兩個階段，
    達到真正的多核心平行。
    """
    def __init__(self, env_fn, num_envs):
        self.num_envs = num_envs
        self.waiting  = False
        self.closed   = False

        # 建立管道對與子進程
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(num_envs)])
        self.procs = []
        for i, (work_remote, remote) in enumerate(zip(self.work_remotes, self.remotes)):
            p = mp.Process(
                target=_worker,
                args=(work_remote, remote, env_fn),
                daemon=True   # 主進程結束時自動清理
            )
            p.start()
            self.procs.append(p)
            work_remote.close()  # 主進程這側不需要子端的 conn

    def reset(self):
        """重置所有環境，回傳 list of states（長度 num_envs）。"""
        for remote in self.remotes:
            remote.send(('reset', None))
        return [remote.recv() for remote in self.remotes]

    def set_config(self, bullet_damage, tie_penalty, use_small_maps=False):
        """同步更新所有子進程的戰鬥參數與地圖池"""
        for remote in self.remotes:
            remote.send(('set_config', (bullet_damage, tie_penalty, use_small_maps)))

    def step_async(self, actions_p1, actions_p2, frame_skip):
        """
        一次把所有動作發送出去（非阻塞），子進程開始平行 step。
        actions_p1, actions_p2 預期為 NumPy arrays
        """
        for i, remote in enumerate(self.remotes):
            remote.send(('step', (actions_p1[i], actions_p2[i], frame_skip)))
        self.waiting = True

    def step_wait(self):
        """
        一起等待所有子進程回傳結果。
        回傳：
          states_list : list[i] = done 前一刻的 state（用於儲存軌跡）
          rewards     : list[i] = (r1, r2)
          dones       : list[i] = bool
          new_states  : list[i] = reset 後的新 state（若 done，否則 None）
        """
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        states_list, rewards, dones, new_states = zip(*results)
        return list(states_list), list(rewards), list(dones), list(new_states)

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()  # 清空未讀取的訊息
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.procs:
            p.join()
        self.closed = True


# ==========================================
# GAE
# ==========================================
def compute_gae(rewards, values, gamma=GAMMA, lam=GAE_LAMBDA):
    advantages = []
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_v = values[t + 1] if t < len(rewards) - 1 else 0.0
        delta  = rewards[t] + gamma * next_v - values[t]
        gae    = delta + gamma * lam * gae
        advantages.insert(0, gae)
    return advantages


def format_time(seconds):
    """將秒數轉換為 HH:MM:SS 格式"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ==========================================
# 主訓練函式
# ==========================================
def train_self_play(resume_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"訓練裝置: {device}")

    model = ConvSNN().to(device)

    total_eps_done = 0
    save_milestone = 2500

    if resume_path and os.path.exists(resume_path):
        logger.info(f"正在從 {resume_path} 載入權重...")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
        try:
            filename = os.path.basename(resume_path)
            if "ep_" in filename:
                ep_num = int(filename.split("ep_")[1].split(".")[0])
                total_eps_done = ep_num
                save_milestone = ((total_eps_done // 2500) + 1) * 2500
                logger.info(f"偵測到進度：已完成 {total_eps_done} 局，下個存檔點：{save_milestone}")
        except Exception:
            pass

    optimizer = optim.Adam(model.parameters(), lr=LR)
    use_amp   = (device.type == "cuda")
    scaler    = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ★ 建立 SubprocVecEnv —— 每個子進程持有獨立的 GameEnv
    logger.info(f"正在啟動 {NUM_ENVS} 個子進程環境（SubprocVecEnv）...")
    from game import GameEnv
    vec_env = VecEnv(env_fn=lambda: GameEnv(render_mode=False), num_envs=NUM_ENVS)

    batch_episodes = max(1, TOTAL_EPS // NUM_ENVS)
    start_batch    = total_eps_done // NUM_ENVS

    logger.info(f"=== PPO + GRU (Envs:{NUM_ENVS}, FrameSkip:{FRAME_SKIP}, "
                f"Epochs:{PPO_EPOCHS}, TBPTT:{TBPTT_LEN}, SubprocVecEnv) ===")

    pbar = tqdm(range(start_batch, batch_episodes), desc="🧠 PPO+GRU", unit="批次", dynamic_ncols=True)

    start_time = time.time()

    for batch_ep in pbar:
        batch_start_time = time.time()
        progress     = batch_ep / max(1, batch_episodes - 1)
        entropy_coef = ENT_START + (ENT_END - ENT_START) * progress

        # 遊戲階段 (Curriculum Learning)
        if total_eps_done < 12500:
            stage_str = "S1(DMG:50, TIE:10, SMALL_MAP)"
            vec_env.set_config(bullet_damage=50, tie_penalty=10.0, use_small_maps=True)
        elif total_eps_done < 20000:
            stage_str = "S2(DMG:35, TIE:30)"
            vec_env.set_config(bullet_damage=35, tie_penalty=30.0, use_small_maps=False)
        else:
            stage_str = "S3(DMG:20, TIE:40)"
            vec_env.set_config(bullet_damage=20, tie_penalty=40.0, use_small_maps=False)

        # ================================================================
        # PHASE 1：ROLLOUT（透過 SubprocVecEnv 並行收集）
        # ================================================================
        model.eval()

        # GRU hidden state：(1, NUM_ENVS, HIDDEN_SIZE)
        h_p1 = torch.zeros(1, NUM_ENVS, HIDDEN_SIZE, device=device)
        h_p2 = torch.zeros(1, NUM_ENVS, HIDDEN_SIZE, device=device)

        # 回合開始：重置所有環境
        states_list = vec_env.reset()

        traj = [
            [{'states': [], 'actions': [], 'rewards': [],
              'values': [], 'old_lp': []} for _ in range(NUM_ENVS)]
            for _ in range(2)
        ]

        # episode_done[i]：此批次中第 i 個環境是否已完成過至少一局
        # 我們只收集每個環境的「第一局」軌跡
        episode_done = [False] * NUM_ENVS

        while not all(episode_done):
            # ── GPU 前向推理 ──
            p1_np = np.stack([s[0] for s in states_list])
            p2_np = np.stack([s[1] for s in states_list])
            p1_batch = torch.as_tensor(p1_np, dtype=torch.float32, device=device)
            p2_batch = torch.as_tensor(p2_np, dtype=torch.float32, device=device)

            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits_p1, vals_p1, h_p1_new = model(p1_batch, h_p1)
                    logits_p2, vals_p2, h_p2_new = model(p2_batch, h_p2)

                logits_p1 = logits_p1.float()
                logits_p2 = logits_p2.float()
                vals_p1   = vals_p1.float().squeeze(-1)
                vals_p2   = vals_p2.float().squeeze(-1)

                dist_p1 = Bernoulli(torch.sigmoid(logits_p1))
                dist_p2 = Bernoulli(torch.sigmoid(logits_p2))
                acts_p1 = dist_p1.sample()   # (NUM_ENVS, 8)
                acts_p2 = dist_p2.sample()
                lp_p1   = dist_p1.log_prob(acts_p1).sum(dim=1)
                lp_p2   = dist_p2.log_prob(acts_p2).sum(dim=1)

            # ★ 更新 hidden state
            h_p1 = h_p1_new.clone()
            h_p2 = h_p2_new.clone()

            # ★ 第一道防線：已完成環境的 hidden state 強制清零
            for i in range(NUM_ENVS):
                if episode_done[i]:
                    h_p1[:, i, :] = 0.0
                    h_p2[:, i, :] = 0.0

            # ── 一次回傳到 CPU NumPy (不要在迴圈裡 cpu()) ──
            acts_p1_np = acts_p1.cpu().numpy()
            acts_p2_np = acts_p2.cpu().numpy()
            vals_p1_np = vals_p1.cpu().numpy()
            vals_p2_np = vals_p2.cpu().numpy()
            lp_p1_np   = lp_p1.cpu().numpy()
            lp_p2_np   = lp_p2.cpu().numpy()

            # ── 儲存「尚未完成」環境的此步軌跡 ──
            for i in range(NUM_ENVS):
                if episode_done[i]:
                    continue
                traj[0][i]['states'].append(torch.from_numpy(p1_np[i]))
                traj[0][i]['actions'].append(torch.from_numpy(acts_p1_np[i]))
                traj[0][i]['values'].append(float(vals_p1_np[i]))
                traj[0][i]['old_lp'].append(float(lp_p1_np[i]))

                traj[1][i]['states'].append(torch.from_numpy(p2_np[i]))
                traj[1][i]['actions'].append(torch.from_numpy(acts_p2_np[i]))
                traj[1][i]['values'].append(float(vals_p2_np[i]))
                traj[1][i]['old_lp'].append(float(lp_p2_np[i]))

            # ── 非同步發送動作（全部子進程開始平行 step）──
            vec_env.step_async(
                acts_p1_np,
                acts_p2_np,
                FRAME_SKIP
            )

            # ── 等待全部子進程回傳結果 ──
            prev_states, rewards_list, step_dones, new_states_list = vec_env.step_wait()

            # ── 彙整結果 ──
            next_states_list = list(states_list)
            for i in range(NUM_ENVS):
                if episode_done[i]:
                    continue   # 這局已結束，忽略後續 step 的資料

                traj[0][i]['rewards'].append(rewards_list[i][0])
                traj[1][i]['rewards'].append(rewards_list[i][1])

                if step_dones[i]:
                    # 這一局結束了
                    episode_done[i] = True
                    # ★ 第二道防線：done 後立刻歸零，確保無殘留
                    h_p1[:, i, :] = 0.0
                    h_p2[:, i, :] = 0.0
                    # Worker 已自動 reset，new_states_list[i] 是新 state
                    next_states_list[i] = new_states_list[i]
                else:
                    next_states_list[i] = prev_states[i]

            states_list = next_states_list

        # ================================================================
        # PHASE 2：GAE + 全局優勢正規化
        # ================================================================
        all_advs_flat = []
        avg_rews_raw  = [[],[]]

        for p in range(2):
            for i in range(NUM_ENVS):
                rews = traj[p][i]['rewards']
                vals = traj[p][i]['values']
                avg_rews_raw[p].append(sum(rews) if rews else 0.0)
                if len(rews) == 0:
                    traj[p][i]['advantages'] = []
                    traj[p][i]['returns']    = []
                    continue
                advs = compute_gae(rews, vals)
                rets = [a + v for a, v in zip(advs, vals)]
                traj[p][i]['advantages'] = advs
                traj[p][i]['returns']    = rets
                all_advs_flat.extend(advs)

        avg_rews = [np.mean(avg_rews_raw[p]) for p in range(2)]

        if len(all_advs_flat) > 1:
            a_mean = float(np.mean(all_advs_flat))
            a_std  = float(np.std(all_advs_flat)) + 1e-8
            for p in range(2):
                for i in range(NUM_ENVS):
                    traj[p][i]['advantages'] = [
                        (a - a_mean) / a_std for a in traj[p][i]['advantages']
                    ]

        # ================================================================
        # PHASE 3 前置：traj → Padding Tensor
        # ================================================================
        T_per_env = [len(traj[0][i]['states']) for i in range(NUM_ENVS)]
        max_T     = max(T_per_env) if max(T_per_env) > 0 else 1

        S_SHAPE = traj[0][0]['states'][0].shape if T_per_env[0] > 0 \
                  else torch.zeros(4, 15, 15).shape
        A_DIM   = 9

        bat_states  = [torch.zeros(max_T, NUM_ENVS, *S_SHAPE) for _ in range(2)]
        bat_actions = [torch.zeros(max_T, NUM_ENVS, A_DIM)    for _ in range(2)]
        bat_old_lp  = [torch.zeros(max_T, NUM_ENVS)           for _ in range(2)]
        bat_adv     = [torch.zeros(max_T, NUM_ENVS)           for _ in range(2)]
        bat_ret     = [torch.zeros(max_T, NUM_ENVS)           for _ in range(2)]
        mask        = torch.zeros(max_T, NUM_ENVS, dtype=torch.bool)

        for i in range(NUM_ENVS):
            T_i = T_per_env[i]
            if T_i == 0:
                continue
            mask[:T_i, i] = True
            for p in range(2):
                bat_states[p][:T_i, i]  = torch.stack(traj[p][i]['states'])
                bat_actions[p][:T_i, i] = torch.stack(traj[p][i]['actions'])
                bat_old_lp[p][:T_i, i] = torch.tensor(traj[p][i]['old_lp'])
                bat_adv[p][:T_i, i]    = torch.tensor(traj[p][i]['advantages'])
                bat_ret[p][:T_i, i]    = torch.tensor(traj[p][i]['returns'])

        bat_states  = [x.to(device) for x in bat_states]
        bat_actions = [x.to(device) for x in bat_actions]
        bat_old_lp  = [x.to(device) for x in bat_old_lp]
        bat_adv     = [x.to(device) for x in bat_adv]
        bat_ret     = [x.to(device) for x in bat_ret]
        mask        = mask.to(device)

        del traj

        # ================================================================
        # PHASE 3：PPO 多 Epoch 更新
        # ================================================================
        model.train()

        for ppo_ep in range(PPO_EPOCHS):
            optimizer.zero_grad()

            h_p1 = torch.zeros(1, NUM_ENVS, HIDDEN_SIZE, device=device)
            h_p2 = torch.zeros(1, NUM_ENVS, HIDDEN_SIZE, device=device)

            ep_losses = []

            for t in range(max_T):
                s1      = bat_states[0][t]
                s2      = bat_states[1][t]
                a1      = bat_actions[0][t]
                a2      = bat_actions[1][t]
                lp1_old = bat_old_lp[0][t]
                lp2_old = bat_old_lp[1][t]
                adv1    = bat_adv[0][t]
                adv2    = bat_adv[1][t]
                ret1    = bat_ret[0][t]
                ret2    = bat_ret[1][t]
                m_t     = mask[t]

                if not m_t.any():
                    with torch.no_grad():
                        _, _, h_p1 = model(s1, h_p1)
                        _, _, h_p2 = model(s2, h_p2)
                    h_p1 = h_p1.detach()
                    h_p2 = h_p2.detach()
                    continue

                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits1, val1, h_p1 = model(s1, h_p1)
                    logits2, val2, h_p2 = model(s2, h_p2)

                logits1 = logits1.float()
                logits2 = logits2.float()
                val1    = val1.float().squeeze(-1)
                val2    = val2.float().squeeze(-1)

                t_loss  = torch.tensor(0.0, device=device)
                valid   = m_t.float()
                n_valid = valid.sum().clamp(min=1)

                for (logits, val, acts, lp_old, adv, ret) in [
                    (logits1, val1, a1, lp1_old, adv1, ret1),
                    (logits2, val2, a2, lp2_old, adv2, ret2),
                ]:
                    probs   = torch.sigmoid(logits)
                    dist_b  = Bernoulli(probs)
                    new_lp  = dist_b.log_prob(acts).sum(dim=1)
                    entropy = dist_b.entropy().sum(dim=1)

                    ratio = torch.exp(new_lp - lp_old)
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv

                    actor_l  = -torch.min(surr1, surr2)
                    critic_l = (val - ret).pow(2)

                    step_loss = actor_l + VALUE_COEF * critic_l - entropy_coef * entropy
                    t_loss    = t_loss + (step_loss * valid).sum() / n_valid

                ep_losses.append(t_loss)

                # ★ TBPTT：截斷梯度
                if (t + 1) % TBPTT_LEN == 0:
                    h_p1 = h_p1.detach()
                    h_p2 = h_p2.detach()

            if len(ep_losses) > 0:
                total_loss = torch.stack(ep_losses).mean()
                scaler.scale(total_loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        del bat_states, bat_actions, bat_old_lp, bat_adv, bat_ret, mask

        # 計算時間資訊
        total_eps_done += NUM_ENVS
        batch_time = time.time() - batch_start_time
        elapsed_time = time.time() - start_time
        
        # 預估剩餘時間 (ETA)
        batches_done = batch_ep - start_batch + 1
        total_batches = batch_episodes - start_batch
        avg_batch_time = elapsed_time / batches_done
        eta_seconds = avg_batch_time * (total_batches - batches_done)
        
        pbar.set_postfix({
            "P1獎勵": f"{avg_rews[0]:.1f}",
            "P2獎勵": f"{avg_rews[1]:.1f}",
            "熵":     f"{entropy_coef:.3f}",
            "局數":   f"{total_eps_done}",
        })

        logger.info(
            f"進度: {stage_str}, 局數={total_eps_done}, "
            f"P1獎勵={avg_rews[0]:.1f}, P2獎勵={avg_rews[1]:.1f}, "
            f"熵={entropy_coef:.3f} | "
            f"耗時: {batch_time:.1f}s, 累計: {format_time(elapsed_time)}, ETA: {format_time(eta_seconds)}"
        )

        if total_eps_done >= save_milestone:
            path = f"snn_ep_{save_milestone}.pth"
            torch.save(model.state_dict(), path)
            logger.info(f"💾 已儲存 {path}（進度：{total_eps_done} 局）")
            save_milestone += 2500

    pbar.close()
    vec_env.close()
    final_path = "snn_ep_final.pth"
    torch.save(model.state_dict(), final_path)
    logger.info(f"🏆 訓練完成！模型已儲存至 {final_path}")


if __name__ == "__main__":
    # ★ Linux fork 通常不需要 spawn，但設定明確較安全
    mp.set_start_method("fork", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint (e.g. snn_ep_5000.pth)")
    args = parser.parse_args()

    train_self_play(resume_path=args.resume)

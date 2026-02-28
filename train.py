import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Bernoulli
import numpy as np
from tqdm import tqdm
from game import GameEnv
from ai import ConvSNN, HIDDEN_SIZE

# ==========================================
# 訓練超參數
# ==========================================
FRAME_SKIP   = 2      # ★ 從 4 降為 2（每 2 幀決策一次）
NUM_ENVS     = 96
GAMMA        = 0.99
GAE_LAMBDA   = 0.95
PPO_EPOCHS   = 4
CLIP_EPS     = 0.2
VALUE_COEF   = 0.5
ENT_START    = 0.10
ENT_END      = 0.01
LR           = 3e-4
MAX_GRAD     = 0.5
TOTAL_EPS    = 5000
TBPTT_LEN    = 20     # 每 TBPTT_LEN 步截斷反向傳播以節省 VRAM


def compute_gae(rewards, values, gamma=GAMMA, lam=GAE_LAMBDA):
    """GAE (廣義優勢估計)：精準把功勞回推給對應步驟"""
    advantages = []
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_v = values[t + 1] if t < len(rewards) - 1 else 0.0
        delta  = rewards[t] + gamma * next_v - values[t]
        gae    = delta + gamma * lam * gae
        advantages.insert(0, gae)
    return advantages


def train_self_play():
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"訓練裝置: {device}")

    model     = ConvSNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    use_amp = (device.type == "cuda")
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"環境準備中，啟動 {NUM_ENVS} 個平行宇宙...")
    envs = [GameEnv(render_mode=False) for _ in range(NUM_ENVS)]

    batch_episodes   = max(1, TOTAL_EPS // NUM_ENVS)
    total_eps_done   = 0
    save_milestone   = 1000

    print(f"=== PPO + GRU (Envs:{NUM_ENVS}, FrameSkip:{FRAME_SKIP}, "
          f"Epochs:{PPO_EPOCHS}, TBPTT:{TBPTT_LEN}) ===")

    pbar = tqdm(range(batch_episodes), desc="🧠 PPO+GRU", unit="批次", dynamic_ncols=True)

    for batch_ep in pbar:
        progress     = batch_ep / max(1, batch_episodes - 1)
        entropy_coef = ENT_START + (ENT_END - ENT_START) * progress

        # ================================================================
        # PHASE 1：ROLLOUT（收集每個環境的完整 Episode 軌跡）
        # ================================================================
        model.eval()

        # 每個環境、每個玩家的 GRU 隱藏狀態：(1, NUM_ENVS, HIDDEN_SIZE)
        h_p1 = torch.zeros(1, NUM_ENVS, HIDDEN_SIZE, device=device)
        h_p2 = torch.zeros(1, NUM_ENVS, HIDDEN_SIZE, device=device)

        states_list = [env.reset() for env in envs]

        # 軌跡儲存：索引 [player][env] = list of values
        traj = [
            [{'states': [], 'actions': [], 'rewards': [],
              'values': [], 'old_lp': []} for _ in range(NUM_ENVS)]
            for _ in range(2)
        ]

        dones = [False] * NUM_ENVS

        while not all(dones):
            # 建立當前批次的狀態張量
            p1_batch = torch.stack([
                torch.tensor(states_list[i][0], dtype=torch.float32)
                for i in range(NUM_ENVS)
            ]).to(device)

            p2_batch = torch.stack([
                torch.tensor(states_list[i][1], dtype=torch.float32)
                for i in range(NUM_ENVS)
            ]).to(device)

            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits_p1, vals_p1, h_p1_new = model(p1_batch, h_p1)
                    logits_p2, vals_p2, h_p2_new = model(p2_batch, h_p2)

                logits_p1 = logits_p1.float()
                logits_p2 = logits_p2.float()
                vals_p1   = vals_p1.float().squeeze(-1)   # (NUM_ENVS,)
                vals_p2   = vals_p2.float().squeeze(-1)

                dist_p1 = Bernoulli(torch.sigmoid(logits_p1))
                dist_p2 = Bernoulli(torch.sigmoid(logits_p2))
                acts_p1 = dist_p1.sample()                # (NUM_ENVS, 8)
                acts_p2 = dist_p2.sample()
                lp_p1   = dist_p1.log_prob(acts_p1).sum(dim=1)  # (NUM_ENVS,)
                lp_p2   = dist_p2.log_prob(acts_p2).sum(dim=1)

            # 更新 hidden states / 已完成的環境歸零
            h_p1 = h_p1_new.clone()
            h_p2 = h_p2_new.clone()

            for i in range(NUM_ENVS):
                if dones[i]:
                    h_p1[:, i, :] = 0.0
                    h_p2[:, i, :] = 0.0

            # 步進每個環境
            next_states_list = list(states_list)
            for i in range(NUM_ENVS):
                if dones[i]:
                    continue

                # 儲存此步軌跡（兩個玩家）
                traj[0][i]['states'].append(p1_batch[i].cpu())
                traj[0][i]['actions'].append(acts_p1[i].cpu())
                traj[0][i]['values'].append(vals_p1[i].item())
                traj[0][i]['old_lp'].append(lp_p1[i].item())

                traj[1][i]['states'].append(p2_batch[i].cpu())
                traj[1][i]['actions'].append(acts_p2[i].cpu())
                traj[1][i]['values'].append(vals_p2[i].item())
                traj[1][i]['old_lp'].append(lp_p2[i].item())

                s, r, d, _ = envs[i].step(
                    acts_p1[i].tolist(), acts_p2[i].tolist(),
                    frame_skip=FRAME_SKIP
                )
                traj[0][i]['rewards'].append(r[0])
                traj[1][i]['rewards'].append(r[1])

                next_states_list[i] = s
                if d:
                    dones[i] = True

            states_list = next_states_list

        # ================================================================
        # PHASE 2：GAE 計算 + 全局正規化
        # ================================================================
        all_advs_flat = []

        for p in range(2):
            for i in range(NUM_ENVS):
                rews = traj[p][i]['rewards']
                vals = traj[p][i]['values']
                if len(rews) == 0:
                    traj[p][i]['advantages'] = []
                    traj[p][i]['returns']    = []
                    continue
                advs = compute_gae(rews, vals)
                rets = [a + v for a, v in zip(advs, vals)]
                traj[p][i]['advantages'] = advs
                traj[p][i]['returns']    = rets
                all_advs_flat.extend(advs)

        # 全局優勢正規化
        if len(all_advs_flat) > 1:
            a_mean = float(np.mean(all_advs_flat))
            a_std  = float(np.std(all_advs_flat)) + 1e-8
            for p in range(2):
                for i in range(NUM_ENVS):
                    traj[p][i]['advantages'] = [
                        (a - a_mean) / a_std for a in traj[p][i]['advantages']
                    ]

        avg_rews = [
            sum(sum(traj[p][i]['rewards']) for i in range(NUM_ENVS)) / NUM_ENVS
            for p in range(2)
        ]

        # ================================================================
        # PHASE 3 前置：把 traj list 轉成 Padding Tensor
        #   shapes: (max_T, NUM_ENVS, ...)
        #   mask  : (max_T, NUM_ENVS) — True = 有效步數
        # ================================================================
        T_per_env = [len(traj[0][i]['states']) for i in range(NUM_ENVS)]
        max_T     = max(T_per_env) if max(T_per_env) > 0 else 1

        # 狀態 shape: (4, 15, 15)
        S_SHAPE = traj[0][0]['states'][0].shape if T_per_env[0] > 0 \
                  else torch.zeros(4, 15, 15).shape
        A_DIM   = 8  # NUM_ACTIONS

        # --- 預先分配 CPU tensor，再一次搬 GPU ---
        # [2 players, max_T, NUM_ENVS, ...]
        bat_states  = [torch.zeros(max_T, NUM_ENVS, *S_SHAPE) for _ in range(2)]
        bat_actions = [torch.zeros(max_T, NUM_ENVS, A_DIM)    for _ in range(2)]
        bat_old_lp  = [torch.zeros(max_T, NUM_ENVS)           for _ in range(2)]
        bat_adv     = [torch.zeros(max_T, NUM_ENVS)           for _ in range(2)]
        bat_ret     = [torch.zeros(max_T, NUM_ENVS)           for _ in range(2)]
        # mask: True = 有效
        mask = torch.zeros(max_T, NUM_ENVS, dtype=torch.bool)

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

        # 一次搬上 GPU
        bat_states  = [x.to(device) for x in bat_states]
        bat_actions = [x.to(device) for x in bat_actions]
        bat_old_lp  = [x.to(device) for x in bat_old_lp]
        bat_adv     = [x.to(device) for x in bat_adv]
        bat_ret     = [x.to(device) for x in bat_ret]
        mask        = mask.to(device)

        # 釋放 traj（已不需要）
        del traj

        # ================================================================
        # PHASE 3：PPO 多 Epoch 更新（跨環境向量化 BPTT）
        #   每個 epoch 只有一個 for t 迴圈，每步同時處理 NUM_ENVS 個環境
        # ================================================================
        model.train()

        for ppo_ep in range(PPO_EPOCHS):
            optimizer.zero_grad()

            # 每個 epoch 都從全零 hidden state 出發（正確 BPTT 起點）
            # shape: (1, NUM_ENVS, HIDDEN_SIZE)
            h_p1 = torch.zeros(1, NUM_ENVS, HIDDEN_SIZE, device=device)
            h_p2 = torch.zeros(1, NUM_ENVS, HIDDEN_SIZE, device=device)

            ep_losses = []   # 收集這個 epoch 的所有有效步損失

            for t in range(max_T):
                # --- 取當前步資料，shape: (NUM_ENVS, ...) ---
                s1 = bat_states[0][t]    # (N, 4, 15, 15)
                s2 = bat_states[1][t]
                a1 = bat_actions[0][t]   # (N, 8)
                a2 = bat_actions[1][t]
                lp1_old = bat_old_lp[0][t]  # (N,)
                lp2_old = bat_old_lp[1][t]
                adv1 = bat_adv[0][t]     # (N,)
                adv2 = bat_adv[1][t]
                ret1 = bat_ret[0][t]     # (N,)
                ret2 = bat_ret[1][t]
                m_t  = mask[t]           # (N,) bool — 有效環境

                if not m_t.any():
                    # 這個 time step 所有環境都已結束，跳過
                    # 但 hidden state 仍需更新（用 no_grad 推進）
                    with torch.no_grad():
                        _, _, h_p1 = model(s1, h_p1)
                        _, _, h_p2 = model(s2, h_p2)
                    h_p1 = h_p1.detach()
                    h_p2 = h_p2.detach()
                    continue

                # --- 批次 Forward（一次算 NUM_ENVS 個環境）---
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits1, val1, h_p1 = model(s1, h_p1)
                    logits2, val2, h_p2 = model(s2, h_p2)

                logits1 = logits1.float()        # (N, 8)
                logits2 = logits2.float()
                val1    = val1.float().squeeze(-1)  # (N,)
                val2    = val2.float().squeeze(-1)

                # --- 計算 Masked PPO Loss ---
                t_loss = torch.tensor(0.0, device=device)
                valid  = m_t.float()             # (N,)  1=有效, 0=padding
                n_valid = valid.sum().clamp(min=1)

                for (logits, val, acts, lp_old, adv, ret) in [
                    (logits1, val1, a1, lp1_old, adv1, ret1),
                    (logits2, val2, a2, lp2_old, adv2, ret2),
                ]:
                    probs   = torch.sigmoid(logits)            # (N, 8)
                    dist_b  = Bernoulli(probs)
                    new_lp  = dist_b.log_prob(acts).sum(dim=1) # (N,)
                    entropy = dist_b.entropy().sum(dim=1)       # (N,)

                    ratio = torch.exp(new_lp - lp_old)         # (N,)
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv

                    actor_l  = -torch.min(surr1, surr2)        # (N,)
                    critic_l = (val - ret).pow(2)              # (N,)  MSE element-wise
                    ent_l    = entropy                          # (N,)

                    step_loss = actor_l + VALUE_COEF * critic_l - entropy_coef * ent_l
                    # Masked 平均：只計算有效環境
                    t_loss = t_loss + (step_loss * valid).sum() / n_valid

                ep_losses.append(t_loss)

                # ★ TBPTT：每 TBPTT_LEN 步截斷 hidden 梯度
                if (t + 1) % TBPTT_LEN == 0:
                    h_p1 = h_p1.detach()
                    h_p2 = h_p2.detach()

            # --- 一次 Backward，更新全部參數 ---
            if len(ep_losses) > 0:
                total_loss = torch.stack(ep_losses).mean()
                scaler.scale(total_loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()   # 清除下一 epoch 用

        del bat_states, bat_actions, bat_old_lp, bat_adv, bat_ret, mask

        # ================================================================
        # 進度更新與存檔
        # ================================================================
        total_eps_done += NUM_ENVS

        pbar.set_postfix({
            "P1獎勵": f"{avg_rews[0]:.1f}",
            "P2獎勵": f"{avg_rews[1]:.1f}",
            "熵":     f"{entropy_coef:.3f}",
            "局數":   f"{total_eps_done}",
        })

        if total_eps_done >= save_milestone:
            path = f"snn_ep_{save_milestone}.pth"
            torch.save(model.state_dict(), path)
            tqdm.write(f"💾 已儲存 {path}（進度：{total_eps_done} 局）")
            save_milestone += 1000

    pbar.close()
    final_path = "snn_ep_final.pth"
    torch.save(model.state_dict(), final_path)
    print(f"🏆 訓練完成！模型已儲存至 {final_path}")


if __name__ == "__main__":
    train_self_play()

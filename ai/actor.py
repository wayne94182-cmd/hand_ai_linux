"""
ai/actor.py — CNN + LSTM Actor
支援 6 通道視覺輸入、22 純量、LSTM 雙狀態、
12 離散 Bernoulli 動作、4 維連續通訊向量（Normal）、
Cross-Attention 隊友通訊接收。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Normal

HIDDEN_SIZE = 256
NUM_ACTIONS_DISCRETE = 12   # Bernoulli 動作
NUM_COMM = 4                # 連續通訊向量維度
IN_CHANNELS = 6
NUM_SCALARS = 22


class ConvLSTM(nn.Module):
    """
    CNN + LSTM Actor，支援：
    1. 6 通道輸入
    2. LSTM 雙狀態 (h, c)
    3. 12 個離散動作（Bernoulli）
    4. 4 維連續通訊向量（Normal distribution）
    5. Cross-Attention 接收隊友通訊
    """

    def __init__(self,
                 in_channels=IN_CHANNELS,
                 num_scalars=NUM_SCALARS,
                 hidden_size=HIDDEN_SIZE,
                 num_actions=NUM_ACTIONS_DISCRETE,
                 num_comm=NUM_COMM):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.num_comm = num_comm

        # ── CNN（與原架構相同，只改 in_channels）──
        self.conv1 = nn.Conv2d(in_channels, 16, 3, 1, 1)
        self.bn1   = nn.GroupNorm(4, 16)
        self.conv2 = nn.Conv2d(16, 32, 3, 2, 1)
        self.bn2   = nn.GroupNorm(8, 32)
        self.conv3 = nn.Conv2d(32, 64, 3, 2, 1)
        self.bn3   = nn.GroupNorm(8, 64)
        # 輸出: (B, 64, 4, 4) → flatten 1024

        self.fc_embed = nn.Linear(1024 + num_scalars, hidden_size)
        self.embed_norm = nn.LayerNorm(hidden_size)

        # ── Cross-Attention 通訊接收器 ──
        # Query: LSTM hidden state (B, hidden_size)
        # Key/Value: 隊友通訊向量 (B, K, num_comm) → 投影到 (B, K, hidden_size)
        self.comm_proj = nn.Linear(num_comm, hidden_size)
        self.attn_q    = nn.Linear(hidden_size, hidden_size)
        self.attn_k    = nn.Linear(hidden_size, hidden_size)
        self.attn_v    = nn.Linear(hidden_size, hidden_size)
        self.attn_out  = nn.Linear(hidden_size, hidden_size)
        # 若無隊友通訊（K=0），輸出全零向量

        # embed + comm_context → 輸入 LSTM 前的融合層
        self.fc_fuse = nn.Linear(hidden_size * 2, hidden_size)
        self.fuse_norm = nn.LayerNorm(hidden_size)

        # ── LSTM ──
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.lstm_norm = nn.LayerNorm(hidden_size)

        # ── 輸出頭 ──
        self.fc_actor = nn.Linear(hidden_size, num_actions)    # 離散動作 logits
        self.fc_comm_mu     = nn.Linear(hidden_size, num_comm) # 通訊 μ
        self.fc_comm_logstd = nn.Linear(hidden_size, num_comm) # 通訊 log σ

    def forward(self, x, scalars, hidden=None, comm_in=None):
        """
        x       : (B, 6, 15, 15)
        scalars : (B, 22)
        hidden  : tuple (h, c) 各 (1, B, 256)，或 None
        comm_in : (B, K, 4) 隊友通訊向量，K=0 時傳 None 或空 tensor

        回傳:
          logits     : (B, 12)   離散動作 logits
          comm_mu    : (B, 4)    通訊分佈期望值（tanh 壓縮到 [-1,1]）
          comm_logstd: (B, 4)    log σ，clamp 到 [-4, 0]
          value_feat : (B, 256)  LSTM 輸出特徵（供 Critic 使用）
          new_hidden : (h, c) 各 (1, B, 256)
        """
        B = x.size(0)
        if hidden is None:
            h0 = torch.zeros(1, B, self.hidden_size, device=x.device, dtype=x.dtype)
            c0 = torch.zeros(1, B, self.hidden_size, device=x.device, dtype=x.dtype)
            hidden = (h0, c0)

        # CNN
        c1   = F.relu(self.bn1(self.conv1(x)))
        c2   = F.relu(self.bn2(self.conv2(c1)))
        c3   = F.relu(self.bn3(self.conv3(c2)))
        flat = c3.view(B, -1)
        embed = F.relu(self.embed_norm(
            self.fc_embed(torch.cat([flat, scalars], dim=-1))
        ))  # (B, 256)

        # Cross-Attention
        if comm_in is not None and comm_in.dim() == 3 and comm_in.size(1) > 0:
            # Query 來自上一步的 h
            prev_h = hidden[0].squeeze(0)                          # (B, 256)
            Q = self.attn_q(prev_h).unsqueeze(1)                   # (B, 1, 256)
            KV = self.comm_proj(comm_in)                           # (B, K, 256)
            K = self.attn_k(KV)
            V = self.attn_v(KV)
            scale = self.hidden_size ** 0.5
            attn_w = torch.softmax(
                torch.bmm(Q, K.transpose(1, 2)) / scale, dim=-1
            )                                                      # (B, 1, K)
            ctx = torch.bmm(attn_w, V).squeeze(1)                 # (B, 256)
            ctx = self.attn_out(ctx)
        else:
            ctx = torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)

        # 融合 embed + context
        fused = F.relu(self.fuse_norm(
            self.fc_fuse(torch.cat([embed, ctx], dim=-1))
        ))  # (B, 256)

        # LSTM
        lstm_in = fused.unsqueeze(1)                             # (B, 1, 256)
        lstm_out, new_hidden = self.lstm(lstm_in, hidden)
        feat = self.lstm_norm(lstm_out.squeeze(1))               # (B, 256)

        # 輸出
        logits      = self.fc_actor(feat)                        # (B, 12)
        comm_mu     = torch.tanh(self.fc_comm_mu(feat))          # (B, 4)，[-1,1]
        comm_logstd = self.fc_comm_logstd(feat).clamp(-4, 0)     # (B, 4)

        return logits, comm_mu, comm_logstd, feat, new_hidden

    def get_action(self, state, hidden=None, comm_in=None, deterministic=False):
        """
        推理用輔助函式（與原 get_action 相容）
        state: (view np.ndarray, scalars np.ndarray)
        回傳: (actions_list[12], comm_vec[4], new_hidden)
        """
        device = next(self.parameters()).device
        view, scalars = state
        x  = torch.tensor(view,    dtype=torch.float32).unsqueeze(0).to(device)
        sc = torch.tensor(scalars, dtype=torch.float32).unsqueeze(0).to(device)

        if hidden is not None:
            hidden = (hidden[0].to(device), hidden[1].to(device))

        if comm_in is not None:
            if not isinstance(comm_in, torch.Tensor):
                comm_in = torch.tensor(comm_in, dtype=torch.float32)
            if comm_in.dim() == 2:
                comm_in = comm_in.unsqueeze(0)  # (1, K, 4)
            comm_in = comm_in.to(device)

        self.eval()
        with torch.no_grad():
            logits, mu, logstd, _, new_hidden = self.forward(x, sc, hidden, comm_in)
        self.train()

        probs = torch.sigmoid(logits[0])
        if deterministic:
            actions = (probs > 0.5).float()
            comm = mu[0]
        else:
            actions = Bernoulli(probs).sample()
            std = torch.exp(logstd[0])
            comm = Normal(mu[0], std).sample().clamp(-1, 1)

        return actions.tolist(), comm.tolist(), new_hidden

    def reset_states(self):
        """向後相容空操作"""
        pass

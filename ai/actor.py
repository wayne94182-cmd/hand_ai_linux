"""
ai/actor.py — CNN + LSTM Actor
支援 6 通道視覺輸入、22 純量、LSTM 雙狀態、
12 離散 Bernoulli 動作、4 維連續通訊向量（Normal）、
Cross-Attention 隊友通訊接收。
支援 seq_mode=True 時一次處理整條時間序列（供 PPO 更新使用）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Normal
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

HIDDEN_SIZE = 256
NUM_ACTIONS_DISCRETE = 16   # Bernoulli 動作
NUM_COMM = 4                # 連續通訊向量維度
IN_CHANNELS = 10
NUM_SCALARS = 25


class ConvLSTM(nn.Module):
    """
    CNN + LSTM Actor，支援：
    1. 10 通道輸入（地形/敵人/隊友/威脅/聲音/安全區/武器/醫療包/手榴彈/彈藥）
    2. LSTM 雙狀態 (h, c)
    3. 16 個離散動作（Bernoulli）
    4. 4 維連續通訊向量（Normal distribution）
    5. Cross-Attention 接收隊友通訊
    6. seq_mode=True 一次處理整條時間序列
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

        # ── CNN（加寬版：32→64→128，Flatten 後 2048）──
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 1, 1)
        self.bn1   = nn.GroupNorm(8, 32)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.bn2   = nn.GroupNorm(8, 64)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.bn3   = nn.GroupNorm(16, 128)
        # 輸出: (B, 128, 4, 4) → flatten 後為 128 * 4 * 4 = 2048

        self.fc_embed = nn.Linear(2048 + num_scalars, hidden_size)
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
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=False)
        self.lstm_norm = nn.LayerNorm(hidden_size)

        # ── 輸出頭 ──
        self.fc_actor = nn.Linear(hidden_size, num_actions)    # 離散動作 logits
        self.fc_comm_mu     = nn.Linear(hidden_size, num_comm) # 通訊 μ
        self.fc_comm_logstd = nn.Linear(hidden_size, num_comm) # 通訊 log σ

    def _cnn_embed(self, x, scalars):
        """CNN + fc_embed，支援任意 batch 維度的輸入。
        x:       (N, C, H, W)
        scalars: (N, num_scalars)
        回傳:    (N, hidden_size)
        """
        # Channels Last 記憶體格式優化（GPU 加速）
        x = x.to(memory_format=torch.channels_last)
        c1   = F.relu(self.bn1(self.conv1(x)))
        c2   = F.relu(self.bn2(self.conv2(c1)))
        c3   = F.relu(self.bn3(self.conv3(c2)))
        flat = c3.reshape(x.size(0), -1)  # reshape 支持非連續張量
        embed = F.relu(self.embed_norm(
            self.fc_embed(torch.cat([flat, scalars], dim=-1))
        ))
        return embed

    @torch.compiler.disable
    def _chunked_cnn_checkpoint(self, x_flat, sc_flat):
        """將編譯困難的迴圈抽離並關閉編譯，避免 Dynamo 發生 AssertionError"""
        N = x_flat.size(0)
        chunk_size = 512  # 配合 ROLLOUT_STEPS 優化記憶體分配
        embed_chunks = []
        for i in range(0, N, chunk_size):
            cx = x_flat[i : i+chunk_size]
            csc = sc_flat[i : i+chunk_size]
            ce = gradient_checkpoint(self._cnn_embed, cx, csc, use_reentrant=False)
            embed_chunks.append(ce)
        return torch.cat(embed_chunks, dim=0)

    def _cross_attention(self, prev_h, comm_in):
        """Cross-Attention 通訊接收。
        prev_h:   (N, hidden_size)
        comm_in:  (N, K, num_comm)  K >= 1（K=0 時由呼叫方補 dummy）
        回傳:     (N, hidden_size)，若原始 K=0 則結果為零
        """
        Q  = self.attn_q(prev_h).unsqueeze(1)       # (N, 1, H)
        KV = self.comm_proj(comm_in)                 # (N, K, H)
        K_ = self.attn_k(KV)
        V_ = self.attn_v(KV)
        scale = self.hidden_size ** 0.5
        attn_w = torch.softmax(
            torch.bmm(Q, K_.transpose(1, 2)) / scale, dim=-1
        )                                             # (N, 1, K)
        ctx = torch.bmm(attn_w, V_).squeeze(1)       # (N, H)
        ctx = self.attn_out(ctx)
        return ctx

    def forward(self, x, scalars, hidden=None, comm_in=None, seq_mode=False):
        """
        seq_mode=False（預設，供 rollout 使用）:
          x:       (B, C, H, W)
          scalars: (B, num_scalars)
          comm_in: (B, K, num_comm) 或 None
          hidden:  (h, c) 各 (1, B, hidden_size) 或 None
          回傳:
            logits:      (B, 16)
            comm_mu:     (B, 4)
            comm_logstd: (B, 4)
            feat:        (B, 256)
            new_hidden:  (h, c) 各 (1, B, 256)

        seq_mode=True（供 PPO 更新使用）:
          x:       (T, B, C, H, W)
          scalars: (T, B, num_scalars)
          comm_in: (T, B, K, num_comm) 或 None
          hidden:  (h, c) 各 (1, B, hidden_size) 或 None
          回傳:
            logits:      (T, B, 16)
            comm_mu:     (T, B, 4)
            comm_logstd: (T, B, 4)
            feat:        (T, B, 256)
            new_hidden:  (h, c) 各 (1, B, 256)
        """
        if seq_mode:
            return self._forward_seq(x, scalars, hidden, comm_in)
        else:
            return self._forward_single(x, scalars, hidden, comm_in)

    def _forward_single(self, x, scalars, hidden, comm_in):
        """原始單步 forward，行為與重構前完全一致。"""
        B = x.size(0)
        if hidden is None:
            h0 = torch.zeros(1, B, self.hidden_size, device=x.device, dtype=x.dtype)
            c0 = torch.zeros(1, B, self.hidden_size, device=x.device, dtype=x.dtype)
            hidden = (h0, c0)

        # CNN（Gradient Checkpointing 節省 VRAM）
        # ⚠️ 必須讓至少一個輸入 requires_grad=True，否則 PyTorch 會略過 Checkpoint 的 backward 且不釋放 VRAM
        x.requires_grad_(True)
        embed = gradient_checkpoint(
            self._cnn_embed, x, scalars, use_reentrant=False
        )  # (B, 256)

        # Cross-Attention（compile 友好：避免 if/else graph break）
        prev_h = hidden[0].squeeze(0)     # (B, 256)
        if comm_in is None or comm_in.dim() != 3 or comm_in.size(1) == 0:
            # 補 dummy，計算後用 mask 清零（tracing 時只走一次）
            _comm = torch.zeros(B, 1, self.num_comm, device=x.device, dtype=x.dtype)
            _has_comm = 0.0
        else:
            _comm = comm_in
            _has_comm = 1.0
        ctx = self._cross_attention(prev_h, _comm)
        ctx = ctx * _has_comm

        # 融合 embed + context
        fused = F.relu(self.fuse_norm(
            self.fc_fuse(torch.cat([embed, ctx], dim=-1))
        ))  # (B, 256)

        # LSTM（單步，batch_first=False → 輸入 (1, B, H)）
        lstm_in = fused.unsqueeze(0)                              # (1, B, 256)
        lstm_out, new_hidden = self.lstm(lstm_in, hidden)
        feat = self.lstm_norm(lstm_out.squeeze(0))                # (B, 256)

        # 輸出
        logits      = self.fc_actor(feat)                         # (B, 12)
        comm_mu     = torch.tanh(self.fc_comm_mu(feat))           # (B, 4)
        comm_logstd = self.fc_comm_logstd(feat).clamp(-4, 0)     # (B, 4)

        return logits, comm_mu, comm_logstd, feat, new_hidden

    @torch.compiler.disable
    def _forward_seq(self, x, scalars, hidden, comm_in):
        """整條時間序列 forward，CNN 平行分塊，LSTM 保持時序正確性。"""
        T, B = x.shape[:2]
        N = T * B

        if hidden is None:
            h0 = torch.zeros(1, B, self.hidden_size, device=x.device, dtype=x.dtype)
            c0 = torch.zeros(1, B, self.hidden_size, device=x.device, dtype=x.dtype)
            hidden = (h0, c0)

        # 1. 攤平 CNN（分塊 Gradient Checkpointing 節省 VRAM，此處可完全平行）
        x_flat  = x.reshape(N, *x.shape[2:])          # (T*B, C, H, W)
        sc_flat = scalars.reshape(N, -1)              # (T*B, num_scalars)
        
        # ⚠️ 啟動 Checkpoint 必須的 flag
        x_flat.requires_grad_(True)
        
        # 分塊處理 CNN
        embed_flat = self._chunked_cnn_checkpoint(x_flat, sc_flat)
        
        # 將 CNN 特徵轉回時序維度 (T, B, hidden_size)
        embed_seq = embed_flat.view(T, B, self.hidden_size)

        # ====== 2. 修正致命 Bug：用 for 迴圈重建 Attention 與 LSTM 的時序依賴 ======
        h_t, c_t = hidden
        lstm_out_list = []
        
        # 判斷是否有有效的通訊張量，並預先準備 dummy 以防 Graph Break
        has_comm = (comm_in is not None and comm_in.dim() == 4 and comm_in.size(2) > 0)
        if has_comm:
            _has_comm_scale = 1.0
            _dummy_comm = None
        else:
            _has_comm_scale = 0.0
            _dummy_comm = torch.zeros(B, 1, self.num_comm, device=x.device, dtype=x.dtype)

        for t in range(T):
            curr_embed = embed_seq[t]              # 當前步的 CNN 特徵 (B, 256)
            prev_h = h_t.squeeze(0)                # 取出上一步「真實」的 h (B, 256)
            
            # Cross-Attention (使用 Dummy + Mask 乘法，防 Graph Break)
            curr_comm = comm_in[t] if has_comm else _dummy_comm
            ctx = self._cross_attention(prev_h, curr_comm) * _has_comm_scale
                
            # 融合特徵
            fused = F.relu(self.fuse_norm(
                self.fc_fuse(torch.cat([curr_embed, ctx], dim=-1))
            ))
            
            # LSTM 單步前向傳播 (要求輸入維度 1, B, H)
            lstm_in = fused.unsqueeze(0)
            out_t, (h_t, c_t) = self.lstm(lstm_in, (h_t, c_t))
            lstm_out_list.append(out_t.squeeze(0)) # 收集這步的結果 (B, 256)
            
        # 將 T 步的結果重新堆疊成 (T, B, 256)
        lstm_out = torch.stack(lstm_out_list, dim=0)

        # 3. LayerNorm + 輸出頭（可再次平行運算）
        feat = self.lstm_norm(lstm_out)                            # (T, B, 256)
        logits      = self.fc_actor(feat)                          # (T, B, self.num_actions)
        comm_mu     = torch.tanh(self.fc_comm_mu(feat))            # (T, B, 4)
        comm_logstd = self.fc_comm_logstd(feat).clamp(-4, 0)       # (T, B, 4)

        return logits, comm_mu, comm_logstd, feat, (h_t, c_t)

    def get_action(self, state, hidden=None, comm_in=None, deterministic=False):
        """
        推理用輔助函式（與原 get_action 相容）
        state: (view np.ndarray, scalars np.ndarray)
        回傳: (actions_list[16], comm_vec[4], new_hidden)
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

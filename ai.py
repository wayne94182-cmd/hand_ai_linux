import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# ConvSNN (實際為 Conv + GRU 架構)
# 移除 SNN LIF 神經元，以 GRU 提供時間記憶
# 輸入：(Batch, 5, 15, 15)  ← 單幀 FOV 局部視野（含隊友層）
# 時間記憶：GRU hidden state (1, Batch, 256)
# 輸出：logits(8), value(1), new_hidden
# ==========================================

HIDDEN_SIZE = 256
NUM_ACTIONS = 9


class ConvSNN(nn.Module):
    """
    卷積 + GRU 網路
    CNN 提取當前幀的空間特徵，GRU 跨時間步保存記憶（取代 SNN 膜電位 + Frame Stacking）。
    輸入 shape: (B, 5, 15, 15) — 單幀 egocentric FOV 視野（5 通道）
    輸出: (logits, value, new_hidden)
    """

    def __init__(self, in_channels=4, num_scalars=9, hidden_size=HIDDEN_SIZE, num_actions=NUM_ACTIONS):
        super(ConvSNN, self).__init__()
        self.hidden_size = hidden_size

        # ──────────────────────────────────────
        # CNN 視覺編碼器 (15×15 → 4×4)
        # ──────────────────────────────────────
        # conv1: (B, 5,  15, 15) → (B, 16, 15, 15)
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.GroupNorm(4, 16)

        # conv2: (B, 16, 15, 15) → (B, 32, 8, 8)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2   = nn.GroupNorm(8, 32)

        # conv3: (B, 32, 8, 8) → (B, 64, 4, 4)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3   = nn.GroupNorm(8, 64)

        flatten_size = 64 * 4 * 4  # = 1024

        # 線性嵌入層
        self.fc_embed = nn.Linear(flatten_size + num_scalars, hidden_size)
        self.embed_norm = nn.LayerNorm(hidden_size)

        # ──────────────────────────────────────
        # GRU 時間記憶層
        # input_size = hidden_size (來自 fc_embed)
        # hidden_size = 256
        # batch_first=True → (B, seq=1, H)
        # ──────────────────────────────────────
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.gru_norm = nn.LayerNorm(hidden_size)

        # ──────────────────────────────────────
        # 輸出頭
        # ──────────────────────────────────────
        self.fc_actor  = nn.Linear(hidden_size, num_actions)
        self.fc_critic = nn.Linear(hidden_size, 1)

    # ──────────────────────────────────────────────
    def forward(self, x, scalars, hidden=None):
        """
        x      : (B, C, H, W)  單幀狀態
        scalars: (B, num_scalars) 純量資訊
        hidden : (1, B, hidden_size) 或 None → 自動初始化為 0
        returns: (logits, value, new_hidden)
                  logits: (B, num_actions)
                  value : (B, 1)
                  new_hidden: (1, B, hidden_size)
        """
        B = x.size(0)
        if hidden is None:
            hidden = torch.zeros(1, B, self.hidden_size,
                                 device=x.device, dtype=x.dtype)

        # CNN 特徵提取
        c1   = F.relu(self.bn1(self.conv1(x)))
        c2   = F.relu(self.bn2(self.conv2(c1)))
        c3   = F.relu(self.bn3(self.conv3(c2)))
        flat = c3.view(B, -1)                      # (B, 1024)

        cat_feat = torch.cat([flat, scalars], dim=-1)
        embed = F.relu(self.embed_norm(self.fc_embed(cat_feat)))        # (B, 256)

        # GRU 時間步（單步）
        gru_in  = embed.unsqueeze(1)               # (B, 1, 256)
        gru_out, new_hidden = self.gru(gru_in, hidden)  # (B, 1, 256), (1, B, 256)
        feat    = self.gru_norm(gru_out.squeeze(1))               # (B, 256)

        # Actor / Critic 頭
        logits = self.fc_actor(feat)               # (B, 9)
        value  = self.fc_critic(feat)              # (B, 1)

        return logits, value, new_hidden

    # ──────────────────────────────────────────────
    def reset_states(self):
        """向後相容：過去用來重置 SNN 膜電位，現在為空操作。
        GRU 的 hidden state 由呼叫端透過 forward() 的 hidden 參數管理。"""
        pass

    # ──────────────────────────────────────────────
    def get_action(self, state, hidden=None, deterministic=False):
        """
        推理時使用的輔助函式。
        state  : tuple (view, scalars)
        hidden : (1, 1, hidden_size) 或 None
        returns: (actions_list, probs, new_hidden)
        """
        device    = next(self.parameters()).device
        dtype     = next(self.parameters()).dtype
        view, scalars = state
        state_t   = torch.tensor(view, dtype=torch.float32).unsqueeze(0).to(device)
        scalars_t = torch.tensor(scalars, dtype=torch.float32).unsqueeze(0).to(device)

        if hidden is not None:
            hidden = hidden.to(device)

        is_train = self.training
        self.eval()
        with torch.no_grad():
            logits, _, new_hidden = self.forward(state_t, scalars_t, hidden)
        if is_train:
            self.train()

        probs = torch.sigmoid(logits[0])

        if deterministic:
            actions = (probs > 0.5).float()
        else:
            actions = torch.bernoulli(probs)

        return actions.tolist(), probs, new_hidden

"""
ai/critic.py — MAPPO 中心化評論家（Team Pooling）
訓練時接收所有 agent 的 encoded state，
在 team 內做 Mean Pooling 後 concat 兩個 team 的向量輸入 MLP。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

CRITIC_HIDDEN = 256


class TeamPoolingCritic(nn.Module):
    """
    MAPPO 中心化評論家，使用 Team Pooling。

    輸入：
      team0_feats: (B, N0, HIDDEN_SIZE)  # team_id=0 的所有 agent 特徵
      team1_feats: (B, N1, HIDDEN_SIZE)  # team_id=1 的所有 agent 特徵
      若某隊只有 1 人（如 stage 0-4），N0=1, N1=0，N1=0 時填零向量

    輸出：
      value: (B,)
    """

    def __init__(self, hidden_size=CRITIC_HIDDEN):
        super().__init__()
        # 兩個 team pool 向量 concat 後輸入
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, team0_feats, team1_feats=None):
        """
        team0_feats: (B, N0, 256) 或 (B, 256)（N0=1 時可省略維度）
        team1_feats: (B, N1, 256) 或 None（無對立隊）
        """
        if team0_feats.dim() == 2:
            team0_feats = team0_feats.unsqueeze(1)
        v0 = team0_feats.mean(dim=1)   # (B, 256)

        if team1_feats is None or team1_feats.numel() == 0:
            v1 = torch.zeros_like(v0)
        else:
            if team1_feats.dim() == 2:
                team1_feats = team1_feats.unsqueeze(1)
            v1 = team1_feats.mean(dim=1)   # (B, 256)

        combined = torch.cat([v0, v1], dim=-1)   # (B, 512)
        return self.mlp(combined).squeeze(-1)     # (B,)

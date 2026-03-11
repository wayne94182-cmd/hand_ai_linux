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
      agent_feats: (B, N, HIDDEN_SIZE)
      team_feats: (B, N, HIDDEN_SIZE)
      opp_feats: (B, M, HIDDEN_SIZE) or None

    輸出：
      value: (B, N)
    """

    def __init__(self, hidden_size=CRITIC_HIDDEN):
        super().__init__()
        # Agent feat + Team mean + Opp mean = 3 * hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, agent_feats, team_feats, opp_feats=None):
        # 💡 解法：宣告新的變數 a_feats, t_feats, o_feats 來承接
        # 絕對不要直接覆寫參數名稱 (agent_feats = ...)，避免 Python 3.12 + Dynamo 編譯崩潰
        a_feats = agent_feats.unsqueeze(1) if agent_feats.dim() == 2 else agent_feats
        t_feats = team_feats.unsqueeze(1) if team_feats.dim() == 2 else team_feats
            
        B, N, H = a_feats.shape
        
        # Team mean
        team_mean = t_feats.mean(dim=1, keepdim=True) # (B, 1, 256)
        team_mean_expanded = team_mean.expand(B, N, H)   # (B, N, 256)
        
        # Opponent mean
        if opp_feats is None or opp_feats.numel() == 0:
            opp_mean_expanded = torch.zeros(B, N, H, device=a_feats.device, dtype=a_feats.dtype)
        else:
            o_feats = opp_feats.unsqueeze(1) if opp_feats.dim() == 2 else opp_feats
            opp_mean = o_feats.mean(dim=1, keepdim=True) # (B, 1, 256)
            opp_mean_expanded = opp_mean.expand(B, N, H)   # (B, N, 256)
        
        combined = torch.cat([a_feats, team_mean_expanded, opp_mean_expanded], dim=-1) # (B, N, 768)
        return self.mlp(combined).squeeze(-1) # (B, N)

"""
ai/comm.py — 通訊向量的抽樣、log_prob 計算、以及 detach 管理
"""
import torch
from torch.distributions import Normal


class CommHandler:
    """通訊向量的抽樣、log_prob 計算、以及 detach 管理"""

    @staticmethod
    def sample(mu, logstd):
        """
        mu    : (B, 4)
        logstd: (B, 4)
        回傳:
          comm_vec : (B, 4)，已 clamp 到 [-1, 1]
          log_prob : (B,)，對 PPO update 使用
        """
        std = torch.exp(logstd)
        dist = Normal(mu, std)
        raw = dist.rsample()                         # reparameterize
        comm_vec = raw.clamp(-1.0, 1.0)
        log_prob = dist.log_prob(raw).sum(dim=-1)    # (B,)
        return comm_vec, log_prob

    @staticmethod
    def to_env(comm_vec):
        """
        丟進環境前 detach + to numpy
        comm_vec: (B, 4) tensor
        回傳: (B, 4) numpy array
        """
        return comm_vec.detach().cpu().numpy()

    @staticmethod
    def old_log_prob(mu_old, logstd_old, comm_vec_tensor):
        """
        用舊的分佈參數計算 log_prob，供 PPO ratio 計算
        """
        std_old = torch.exp(logstd_old)
        dist = Normal(mu_old, std_old)
        return dist.log_prob(comm_vec_tensor).sum(dim=-1)

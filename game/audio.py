"""
game/audio.py — 聲音波紋系統
每次事件（腳步、換彈、槍聲、爆炸）產生 SoundWave，
擴散後投影到觀測者的局部視野 Channel。
"""
import math
import numpy as np
from dataclasses import dataclass

from game.config import TILE_SIZE, VIEW_SIZE, VIEW_CENTER


@dataclass
class SoundWave:
    x: float
    y: float
    current_radius: float        # 目前波紋半徑（pixel）
    max_radius: float            # 最大傳播半徑後消滅
    sound_value: float           # 0.3=腳步, 0.6=換彈/打藥, 1.0=槍聲/爆炸
    expand_speed: float = 90.0   # pixel/frame，約 2.25 格/frame

    def update(self):
        self.current_radius += self.expand_speed

    def alive(self) -> bool:
        return self.current_radius < self.max_radius


# ─── 工廠函式（由 env 呼叫）──────────────────────────────

def create_footstep_wave(x: float, y: float) -> SoundWave:
    return SoundWave(x, y, 0.0, 240.0, 0.3)

def create_reload_wave(x: float, y: float) -> SoundWave:
    return SoundWave(x, y, 0.0, 360.0, 0.6)

def create_gunshot_wave(x: float, y: float) -> SoundWave:
    return SoundWave(x, y, 0.0, 600.0, 1.0)

def create_explosion_wave(x: float, y: float) -> SoundWave:
    return SoundWave(x, y, 0.0, 960.0, 1.0)


# ─── 雙線性插值（與 env._inject_value 邏輯相同）─────────

def _inject_value(channel: np.ndarray, r_f: float, c_f: float,
                  value: float, size: int) -> None:
    r0 = int(math.floor(r_f))
    c0 = int(math.floor(c_f))
    dr = r_f - r0
    dc = c_f - c0
    if 0 <= r0 < size and 0 <= c0 < size:
        channel[r0, c0] += value * (1.0 - dr) * (1.0 - dc)
    if 0 <= r0 < size and 0 <= c0 + 1 < size:
        channel[r0, c0 + 1] += value * (1.0 - dr) * dc
    if 0 <= r0 + 1 < size and 0 <= c0 < size:
        channel[r0 + 1, c0] += value * dr * (1.0 - dc)
    if 0 <= r0 + 1 < size and 0 <= c0 + 1 < size:
        channel[r0 + 1, c0 + 1] += value * dr * dc


# ─── 核心渲染函式 ────────────────────────────────────────

def render_sound_channel(
    observer_x: float, observer_y: float,
    fwd_x: float, fwd_y: float,
    rgt_x: float, rgt_y: float,
    sound_waves: list,
    view_size: int = VIEW_SIZE,
    view_center: float = VIEW_CENTER,
    tile_size: float = TILE_SIZE,
) -> np.ndarray:
    """
    把所有 SoundWave 的波紋弧線投影到 observer 的局部視野上。

    演算法：
    1. 對每個 SoundWave，計算波紋弧線上的取樣點（以 wave.x/y 為圓心，
       current_radius 為半徑，每隔 5 度取一個點）。
    2. 把每個取樣點轉換到 observer 的局部座標系（fwd/rgt 投影）。
    3. 若該點落在 view_size×view_size 範圍內，
       在對應格子累加 sound_value（用 _inject_value 雙線性插值）。
    4. 最終對整個 channel 做 np.clip(0, 1)。

    注意：波紋是「環形」，只有 current_radius ± 1格 的薄殼才有值，
    不是填滿的圓。因此取樣點只取邊緣，不取內部。

    回傳 shape = (view_size, view_size), dtype=float32
    """
    channel = np.zeros((view_size, view_size), dtype=np.float32)

    for wave in sound_waves:
        if not wave.alive() or wave.current_radius <= 0:
            continue

        r = wave.current_radius
        # 圓周取樣，每 5 度一個點 → 72 個點
        for deg in range(0, 360, 5):
            rad = math.radians(deg)
            px = wave.x + math.cos(rad) * r
            py = wave.y + math.sin(rad) * r

            # 轉換到 observer 的局部座標系
            dx = px - observer_x
            dy = py - observer_y
            ft = (dx * fwd_x + dy * fwd_y) / tile_size
            rt = (dx * rgt_x + dy * rgt_y) / tile_size

            # 45 度旋轉投影（與 env 中 enemy/projectile 的投影一致）
            dr = (ft + rt) / 1.41421356
            dc = (ft - rt) / 1.41421356

            r_f = view_center + dr
            c_f = view_center + dc

            if 0 <= r_f < view_size and 0 <= c_f < view_size:
                _inject_value(channel, r_f, c_f, wave.sound_value, view_size)

    np.clip(channel, 0.0, 1.0, out=channel)
    return channel

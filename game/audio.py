"""
game/audio.py — 聲音波紋系統
每次事件（腳步、換彈、槍聲、爆炸）產生 SoundWave，
擴散後投影到觀測者的局部視野 Channel。
修正：使用 birth_frame + 逐幀掃描解決 frame_skip 跳格問題。
Numba 優化：使用 JIT 編譯加速核心渲染循環。
"""
import math
import numpy as np
from dataclasses import dataclass
from numba import njit

from game.config import TILE_SIZE, VIEW_SIZE, VIEW_CENTER, AudioConfig


@dataclass
class SoundWave:
    x: float
    y: float
    birth_frame: int             # 出生時的 frame_count
    max_radius: float            # 最大傳播半徑後消滅
    sound_value: float           # 0.3=腳步, 0.6=換彈/打藥, 1.0=槍聲/爆炸
    expand_speed: float = AudioConfig.FOOTSTEP_EXPAND_SPEED   # pixel/frame，約 2.25 格/frame

    def radius_at(self, frame: int) -> float:
        return (frame - self.birth_frame) * self.expand_speed

    def alive(self, frame: int) -> bool:
        return self.radius_at(frame) < self.max_radius


# ─── 工廠函式（由 env 呼叫，傳入 frame_count）──────────

def create_footstep_wave(x: float, y: float, frame: int) -> SoundWave:
    return SoundWave(x, y, birth_frame=frame, max_radius=AudioConfig.FOOTSTEP_MAX_RADIUS, sound_value=0.3)

def create_reload_wave(x: float, y: float, frame: int) -> SoundWave:
    return SoundWave(x, y, birth_frame=frame, max_radius=AudioConfig.RELOAD_MAX_RADIUS, sound_value=0.6)

def create_gunshot_wave(x: float, y: float, frame: int) -> SoundWave:
    return SoundWave(x, y, birth_frame=frame, max_radius=AudioConfig.GUNSHOT_MAX_RADIUS, sound_value=1.0)

def create_explosion_wave(x: float, y: float, frame: int) -> SoundWave:
    return SoundWave(x, y, birth_frame=frame, max_radius=AudioConfig.EXPLOSION_MAX_RADIUS, sound_value=1.0)


# ─── Numba 編譯的雙線性插值 ─────────

@njit(cache=True, fastmath=True)
def _inject_value_njit(channel, r_f, c_f, value, size):
    """Numba 編譯的雙線性插值"""
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


# ─── Numba 編譯的核心渲染循環 ────────────────────────

@njit(cache=True, fastmath=True)
def _render_sound_waves_njit(
    channel, wave_xs, wave_ys, wave_births, wave_max_radii, wave_values, wave_expand_speeds,
    observer_x, observer_y, fwd_x, fwd_y, rgt_x, rgt_y,
    current_frame, prev_frame, view_size, view_center, tile_size
):
    """Numba 編譯的聲音波紋渲染核心循環"""
    for i in range(len(wave_xs)):
        wave_x = wave_xs[i]
        wave_y = wave_ys[i]
        birth_frame = wave_births[i]
        max_radius = wave_max_radii[i]
        sound_value = wave_values[i]
        expand_speed = wave_expand_speeds[i]

        for f in range(prev_frame + 1, current_frame + 1):
            r = (f - birth_frame) * expand_speed
            if r <= 0 or r >= max_radius:
                continue

            # 圓周取樣，每 5 度一個點 → 72 個點
            for deg in range(0, 360, 5):
                rad = math.radians(float(deg))
                px = wave_x + math.cos(rad) * r
                py = wave_y + math.sin(rad) * r

                # 轉換到 observer 的局部座標系
                dx = px - observer_x
                dy = py - observer_y
                ft = (dx * fwd_x + dy * fwd_y) / tile_size
                rt = (dx * rgt_x + dy * rgt_y) / tile_size

                # 45 度旋轉投影
                dr_v = (ft + rt) / 1.41421356
                dc_v = (ft - rt) / 1.41421356

                r_f = view_center + dr_v
                c_f = view_center + dc_v

                if 0 <= r_f < view_size and 0 <= c_f < view_size:
                    _inject_value_njit(channel, r_f, c_f, sound_value, view_size)


# ─── 公開介面（Python 包裝） ────────────────────────────

def render_sound_channel(
    observer_x: float, observer_y: float,
    fwd_x: float, fwd_y: float,
    rgt_x: float, rgt_y: float,
    sound_waves: list,
    current_frame: int,
    prev_frame: int,
    view_size: int = VIEW_SIZE,
    view_center: float = VIEW_CENTER,
    tile_size: float = TILE_SIZE,
) -> np.ndarray:
    """
    把所有 SoundWave 的波紋弧線投影到 observer 的局部視野上。

    修正版：對 [prev_frame+1 ... current_frame] 每個整數幀計算半徑並投影，
    避免 frame_skip 時波紋跳格漏失。
    Numba 優化版：使用 JIT 編譯加速核心循環。

    回傳 shape = (view_size, view_size), dtype=float32
    """
    channel = np.zeros((view_size, view_size), dtype=np.float32)

    if not sound_waves:
        return channel

    # 批次提取波紋數據到 NumPy 數組（Numba 友好格式）
    wave_xs = np.array([w.x for w in sound_waves], dtype=np.float32)
    wave_ys = np.array([w.y for w in sound_waves], dtype=np.float32)
    wave_births = np.array([w.birth_frame for w in sound_waves], dtype=np.int32)
    wave_max_radii = np.array([w.max_radius for w in sound_waves], dtype=np.float32)
    wave_values = np.array([w.sound_value for w in sound_waves], dtype=np.float32)
    wave_expand_speeds = np.array([w.expand_speed for w in sound_waves], dtype=np.float32)

    # 調用 Numba 編譯的核心函數
    _render_sound_waves_njit(
        channel, wave_xs, wave_ys, wave_births, wave_max_radii, wave_values, wave_expand_speeds,
        float(observer_x), float(observer_y), float(fwd_x), float(fwd_y), float(rgt_x), float(rgt_y),
        current_frame, prev_frame, view_size, float(view_center), float(tile_size)
    )

    np.clip(channel, 0.0, 1.0, out=channel)
    return channel

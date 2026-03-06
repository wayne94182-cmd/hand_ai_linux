import math

import numpy as np
from numba import njit

from game.config import (
    TILE_SIZE, COLS, ROWS,
    VIEW_SIZE, VIEW_CENTER, VIEW_RANGE,
    FOV_DEGREES, HALF_FOV,
)

# ─── 狙擊視野常數 ───
SNIPER_TILE_SIZE = 80       # 縮放後一格代表兩格實際距離
SNIPER_FOV_DEGREES = 60.0   # 狙擊鏡視角較窄
SNIPER_HALF_FOV = 30.0
SNIPER_VIEW_RANGE = 10      # 格數不變，但因 TILE_SIZE×2，實際距離翻倍


def _precompute_fov(
    view_size=VIEW_SIZE, view_center=VIEW_CENTER,
    view_range=VIEW_RANGE, fov_degrees=FOV_DEGREES, half_fov=HALF_FOV
) -> tuple:
    rows_idx = np.arange(view_size)
    cols_idx = np.arange(view_size)
    col_g, row_g = np.meshgrid(cols_idx, rows_idx)

    # 計算相對距離 (向下為正 dr，向右為正 dc)
    dr = (row_g - view_center).astype(np.float32)
    dc = (col_g - view_center).astype(np.float32)

    # 135度旋轉投影 (面向右下角)
    fwd = (dr + dc) / 1.41421356
    right = (dr - dc) / 1.41421356

    dist = np.hypot(fwd, right)
    angle = np.degrees(np.arctan2(right, fwd))
    in_fov = (dist <= view_range) & ((dist == 0) | (np.abs(angle) <= half_fov))

    fov_rc = list(zip(*np.where(in_fov)))
    n = len(fov_rc)
    fov_rc_np = np.array(fov_rc, dtype=np.int32)
    fov_fwd = np.array([fwd[r, c] for r, c in fov_rc], dtype=np.float32)
    fov_right = np.array([right[r, c] for r, c in fov_rc], dtype=np.float32)

    ray_samples_list = []
    for (r, c) in fov_rc:
        ft = float(fov_fwd[len(ray_samples_list)])
        rt = float(fov_right[len(ray_samples_list)])
        d = float(dist[r, c])
        if d <= 0:
            ray_samples_list.append(np.empty((0, 2), dtype=np.float32))
        else:
            m = max(2, int(d / 0.4) + 2)
            ts = np.linspace(0.0, 1.0, m)[1:-1]
            pts = np.column_stack([ft * ts, rt * ts]).astype(np.float32)
            ray_samples_list.append(pts)

    ray_offsets = np.zeros(n, dtype=np.int32)
    ray_lengths = np.zeros(n, dtype=np.int32)
    total_pts = sum(len(pts) for pts in ray_samples_list)
    ray_flat = np.zeros((total_pts, 2), dtype=np.float32)

    idx = 0
    for k, pts in enumerate(ray_samples_list):
        l = len(pts)
        ray_offsets[k] = idx
        ray_lengths[k] = l
        if l > 0:
            ray_flat[idx : idx + l] = pts
        idx += l

    return fov_rc_np, fov_fwd, fov_right, ray_flat, ray_offsets, ray_lengths


# ─── 一般視野預計算（預設參數）───
_FOV_RC_NP, _FOV_FWD, _FOV_RIGHT, _RAY_FLAT, _RAY_OFFSETS, _RAY_LENGTHS = _precompute_fov()

# ─── 狙擊視野預計算 ───
_SNIPER_FOV_RC_NP, _SNIPER_FOV_FWD, _SNIPER_FOV_RIGHT, \
    _SNIPER_RAY_FLAT, _SNIPER_RAY_OFFSETS, _SNIPER_RAY_LENGTHS = _precompute_fov(
        view_size=VIEW_SIZE, view_center=VIEW_CENTER,
        view_range=SNIPER_VIEW_RANGE,
        fov_degrees=SNIPER_FOV_DEGREES, half_fov=SNIPER_HALF_FOV,
    )


def get_fov_tables(sniper=False) -> tuple:
    """回傳對應的六個 FOV 預計算陣列。"""
    if sniper:
        return (
            _SNIPER_FOV_RC_NP, _SNIPER_FOV_FWD, _SNIPER_FOV_RIGHT,
            _SNIPER_RAY_FLAT, _SNIPER_RAY_OFFSETS, _SNIPER_RAY_LENGTHS,
        )
    return (
        _FOV_RC_NP, _FOV_FWD, _FOV_RIGHT,
        _RAY_FLAT, _RAY_OFFSETS, _RAY_LENGTHS,
    )


@njit(cache=True)
def njit_has_line_of_sight(x1, y1, x2, y2, grid_np, tile_size, cols, rows):
    dx = x2 - x1
    dy = y2 - y1
    dist = math.sqrt(dx * dx + dy * dy)
    if dist <= 0.0:
        return True
    steps = max(1, int(dist / (tile_size * 0.5)))
    sx = dx / steps
    sy = dy / steps
    cx = x1
    cy = y1
    prev_tc = int(cx // tile_size)
    prev_tr = int(cy // tile_size)
    for _ in range(steps - 1):
        cx += sx
        cy += sy
        tc = int(cx // tile_size)
        tr = int(cy // tile_size)
        if 0 <= tc < cols and 0 <= tr < rows:
            if grid_np[tr, tc] == 1:
                return False
            if tc != prev_tc and tr != prev_tr:
                b_wall = (0 <= prev_tc < cols and 0 <= tr < rows and grid_np[tr, prev_tc] == 1)
                c_wall = (0 <= tc < cols and 0 <= prev_tr < rows and grid_np[prev_tr, tc] == 1)
                if b_wall and c_wall:
                    return False
        else:
            return False
        prev_tc = tc
        prev_tr = tr
    return True


# ─── 通用 FOV 計算核心（接受 tables 作為參數）───

@njit(cache=True)
def njit_compute_fov(
    ax, ay, fwd_x, fwd_y, rgt_x, rgt_y, grid_np,
    fov_rc_np, fov_fwd, fov_right, ray_flat, ray_offsets, ray_lengths,
    tile_size, cols, rows, view_size,
):
    ch0 = np.full((view_size, view_size), -1.0, dtype=np.float32)
    n = fov_rc_np.shape[0]

    for k in range(n):
        r_idx = fov_rc_np[k, 0]
        c_idx = fov_rc_np[k, 1]
        ft = fov_fwd[k]
        rt = fov_right[k]

        wx = ax + fwd_x * ft * tile_size + rgt_x * rt * tile_size
        wy = ay + fwd_y * ft * tile_size + rgt_y * rt * tile_size

        tc = int(wx // tile_size)
        tr_v = int(wy // tile_size)

        if not (0 <= tc < cols and 0 <= tr_v < rows):
            ch0[r_idx, c_idx] = 1.0
            continue

        off = ray_offsets[k]
        l = ray_lengths[k]
        blocked = False
        prev_ic = int(ax // tile_size)
        prev_ir = int(ay // tile_size)
        for j in range(l):
            fft = ray_flat[off + j, 0]
            rrt = ray_flat[off + j, 1]
            ix = ax + fwd_x * fft * tile_size + rgt_x * rrt * tile_size
            iy = ay + fwd_y * fft * tile_size + rgt_y * rrt * tile_size
            ic = int(ix // tile_size)
            ir = int(iy // tile_size)
            if 0 <= ic < cols and 0 <= ir < rows:
                if grid_np[ir, ic] == 1:
                    blocked = True
                    break
                if ic != prev_ic and ir != prev_ir:
                    b_wall = (0 <= prev_ic < cols and 0 <= ir < rows and grid_np[ir, prev_ic] == 1)
                    c_wall = (0 <= ic < cols and 0 <= prev_ir < rows and grid_np[prev_ir, ic] == 1)
                    if b_wall and c_wall:
                        blocked = True
                        break
            else:
                blocked = True
                break
            prev_ic = ic
            prev_ir = ir

        if not blocked:
            ch0[r_idx, c_idx] = float(grid_np[tr_v, tc])

    return ch0


# ─── 專用函式：各自帶入預綁定的 tables，Numba 只編譯一次 ───

@njit(cache=True)
def _njit_compute_fov_core(
    ax, ay, fwd_x, fwd_y, rgt_x, rgt_y, grid_np,
    fov_rc_np, fov_fwd, fov_right, ray_flat, ray_offsets, ray_lengths,
    tile_size, cols, rows, view_size,
):
    ch0 = np.full((view_size, view_size), -1.0, dtype=np.float32)
    n = fov_rc_np.shape[0]

    for k in range(n):
        r_idx = fov_rc_np[k, 0]
        c_idx = fov_rc_np[k, 1]
        ft = fov_fwd[k]
        rt = fov_right[k]

        wx = ax + fwd_x * ft * tile_size + rgt_x * rt * tile_size
        wy = ay + fwd_y * ft * tile_size + rgt_y * rt * tile_size

        tc = int(wx // tile_size)
        tr_v = int(wy // tile_size)

        if not (0 <= tc < cols and 0 <= tr_v < rows):
            ch0[r_idx, c_idx] = 1.0
            continue

        off = ray_offsets[k]
        l = ray_lengths[k]
        blocked = False
        prev_ic = int(ax // tile_size)
        prev_ir = int(ay // tile_size)
        for j in range(l):
            fft = ray_flat[off + j, 0]
            rrt = ray_flat[off + j, 1]
            ix = ax + fwd_x * fft * tile_size + rgt_x * rrt * tile_size
            iy = ay + fwd_y * fft * tile_size + rgt_y * rrt * tile_size
            ic = int(ix // tile_size)
            ir = int(iy // tile_size)
            if 0 <= ic < cols and 0 <= ir < rows:
                if grid_np[ir, ic] == 1:
                    blocked = True
                    break
                if ic != prev_ic and ir != prev_ir:
                    b_wall = (0 <= prev_ic < cols and 0 <= ir < rows and grid_np[ir, prev_ic] == 1)
                    c_wall = (0 <= ic < cols and 0 <= prev_ir < rows and grid_np[prev_ir, ic] == 1)
                    if b_wall and c_wall:
                        blocked = True
                        break
            else:
                blocked = True
                break
            prev_ic = ic
            prev_ir = ir

        if not blocked:
            ch0[r_idx, c_idx] = float(grid_np[tr_v, tc])

    return ch0


def njit_compute_fov_standard(ax, ay, fwd_x, fwd_y, rgt_x, rgt_y,
                               grid_np, tile_size, cols, rows, view_size):
    """標準 FOV — 使用預綁定的 standard tables"""
    return _njit_compute_fov_core(
        ax, ay, fwd_x, fwd_y, rgt_x, rgt_y, grid_np,
        _FOV_RC_NP, _FOV_FWD, _FOV_RIGHT,
        _RAY_FLAT, _RAY_OFFSETS, _RAY_LENGTHS,
        tile_size, cols, rows, view_size,
    )


def njit_compute_fov_sniper(ax, ay, fwd_x, fwd_y, rgt_x, rgt_y,
                             grid_np, tile_size, cols, rows, view_size):
    """狙擊 FOV — 使用預綁定的 sniper tables"""
    return _njit_compute_fov_core(
        ax, ay, fwd_x, fwd_y, rgt_x, rgt_y, grid_np,
        _SNIPER_FOV_RC_NP, _SNIPER_FOV_FWD, _SNIPER_FOV_RIGHT,
        _SNIPER_RAY_FLAT, _SNIPER_RAY_OFFSETS, _SNIPER_RAY_LENGTHS,
        tile_size, cols, rows, view_size,
    )

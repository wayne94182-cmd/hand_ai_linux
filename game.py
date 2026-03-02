import pygame
import math
import random
import numpy as np
from numba import njit

# ==========================================
# 遊戲設定
# ==========================================
TILE_SIZE = 40
COLS, ROWS = 32, 24
WIDTH, HEIGHT = COLS * TILE_SIZE, ROWS * TILE_SIZE
FPS = 60

# AI 視野設定
VIEW_SIZE   = 15
VIEW_CENTER = VIEW_SIZE // 2   # = 7
FOV_DEGREES = 130.0
HALF_FOV    = FOV_DEGREES / 2.0
VIEW_RANGE  = 14               # 格數

NUM_ACTIONS = 9
MAX_FRAMES  = 1200

MAX_FRAMES  = 1200

# ==========================================
# 獎懲機制與階段設定
# ==========================================
class GameConfig:
    # 行為成本與獎勵
    SHOOT_PENALTY   = 0.04    # 開槍成本
    RADAR_REWARD    = 0.002   # 敵人在雷達內的每幀獎勵
    SURVIVAL_COST   = 0.01    # 每幀存活成本
    DASH_COST_HP    = 5       # 衝刺扣除血量
    DASH_PENALTY    = -1.0    # 衝刺扣分
    REGEN_AMOUNT    = 20      # 靜止回血量

    # 戰鬥數值
    HIT_REWARD      = 10.0    # 擊中對手獎勵
    DAMAGE_PENALTY  = 10.0    # 被擊中懲罰
    WIN_REWARD      = 100.0   # 勝利獎勵
    LOSE_PENALTY    = 30.0    # 失敗懲罰

    # 階段性變動參數 (可被外部覆蓋)
    BULLET_DAMAGE   = 20      # 子彈傷害
    TIE_PENALTY     = 10.0    # 雙方皆存活的平手懲罰

# ==========================================
# 地圖
# ==========================================
# ==========================================
# 地圖 (使用 NumPy 加速系統效率)
# ==========================================
def create_base_grid():
    """建立帶有邊界牆壁的基礎地圖"""
    m = np.zeros((ROWS, COLS), dtype=np.int8)
    m[0, :] = 1; m[-1, :] = 1; m[:, 0] = 1; m[:, -1] = 1
    return m

# --- 原有地圖 NumPy 向量化 ---
MAP_TINY_1 = np.ones((ROWS, COLS), dtype=np.int8)
MAP_TINY_1[6:18, 8:24] = 0
MAP_TINY_1[11:13, 14:16] = 1
MAP_TINY_1[11, 17:19] = 1

MAP_OPEN = create_base_grid()
for (rr, cc) in [(6,8),(6,24),(17,8),(17,24),(11,16),(12,16)]:
    MAP_OPEN[rr, cc] = 1

MAP_SMALL_COVER = create_base_grid()
for (rr, cc) in [
    (5,5),(5,6),(6,5),(5,26),(5,25),(6,26),
    (18,5),(18,6),(17,5),(18,26),(18,25),(17,26),
    (11,14),(11,15),(11,16),(12,14),(12,15),(12,16),
]:
    MAP_SMALL_COVER[rr, cc] = 1

MAP_1 = np.array([
    [1]*32,
    [1]+[0]*30+[1],[1]+[0]*30+[1],
    [1,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,1],
    [1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1]+[0]*30+[1],[1]+[0]*30+[1],[1]+[0]*30+[1],[1]+[0]*30+[1],
    [1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1],
    [1,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,1],
    [1]+[0]*30+[1],[1]+[0]*30+[1],[1]+[0]*30+[1],[1]*32,
], dtype=np.int8)

# --- 新增 4 個不同尺寸的小地圖 ---
MAP_SMALL_1 = np.ones((ROWS, COLS), dtype=np.int8)
MAP_SMALL_1[5:19, 7:25] = 0 # 14x18 矩形空間

MAP_SMALL_2 = np.ones((ROWS, COLS), dtype=np.int8)
MAP_SMALL_2[3:21, 10:22] = 0 # 18x12 窄長空間

MAP_SMALL_3 = np.ones((ROWS, COLS), dtype=np.int8)
MAP_SMALL_3[4:14, 4:14] = 0 # 左上室
MAP_SMALL_3[10:20, 18:28] = 0 # 右下室
MAP_SMALL_3[10:14, 14:18] = 0 # 連接通道

MAP_SMALL_4 = np.ones((ROWS, COLS), dtype=np.int8)
MAP_SMALL_4[6:18, 4:28] = 0 # 寬扁空間
MAP_SMALL_4[11:13, 10] = 1; MAP_SMALL_4[11:13, 21] = 1 # 柱子

# --- 新增 3 個空曠的大地圖 ---
MAP_LARGE_1 = create_base_grid() # 四角柱子
MAP_LARGE_1[5:9, 7:11] = 1; MAP_LARGE_1[5:9, 21:25] = 1
MAP_LARGE_1[15:19, 7:11] = 1; MAP_LARGE_1[15:19, 21:25] = 1

MAP_LARGE_2 = create_base_grid() # 中央寬敞十字遮蔽
MAP_LARGE_2[11:13, 6:14] = 1; MAP_LARGE_2[11:13, 18:26] = 1
MAP_LARGE_2[6:10, 15:17] = 1; MAP_LARGE_2[14:18, 15:17] = 1

MAP_LARGE_3 = create_base_grid() # 散落的小方塊
for r in [5, 12, 18]:
    for c in [8, 16, 24]:
        if (r, c) != (12, 16): # 留空中央
            MAP_LARGE_3[r:r+1, c:c+1] = 1

# 地圖池更新
TINY_MAPS = [MAP_TINY_1, MAP_OPEN, MAP_SMALL_COVER, MAP_SMALL_1, MAP_SMALL_2, MAP_SMALL_3, MAP_SMALL_4]
MAPS = TINY_MAPS + [MAP_1, MAP_LARGE_1, MAP_LARGE_2, MAP_LARGE_3]
SMALL_MAP_IDS = {id(m) for m in TINY_MAPS}



# ==========================================
# FOV 視野快取（含 Numba 用的壓扁格式）
# ==========================================
def _precompute_fov():
    rows_idx = np.arange(VIEW_SIZE)
    cols_idx = np.arange(VIEW_SIZE)
    col_g, row_g = np.meshgrid(cols_idx, rows_idx)
    fwd   = (VIEW_CENTER - row_g).astype(np.float32)
    right = (col_g - VIEW_CENTER).astype(np.float32)
    dist  = np.hypot(fwd, right)
    angle = np.degrees(np.arctan2(right, fwd))
    in_fov = (dist <= VIEW_RANGE) & ((dist == 0) | (np.abs(angle) <= HALF_FOV))

    fov_rc = list(zip(*np.where(in_fov)))   # list of (row, col)
    N = len(fov_rc)

    # ── Numba 用：把 fov_rc 轉成 int32 array ──
    fov_rc_np = np.array(fov_rc, dtype=np.int32)       # (N, 2)
    fov_fwd   = np.array([fwd[r, c]   for r, c in fov_rc], dtype=np.float32)
    fov_right = np.array([right[r, c] for r, c in fov_rc], dtype=np.float32)

    # ── 原本的 ray_samples（Python list，供非 Numba 路徑備用）──
    ray_samples_list = []
    for (r, c) in fov_rc:
        ft = float(fwd[r, c])
        rt = float(right[r, c])
        d  = float(dist[r, c])
        if d <= 0:
            ray_samples_list.append(np.empty((0, 2), dtype=np.float32))
        else:
            n  = max(2, int(d / 0.4) + 2)
            ts = np.linspace(0.0, 1.0, n)[1:-1]
            pts = np.column_stack([ft * ts, rt * ts]).astype(np.float32)
            ray_samples_list.append(pts)

    # ── ★ Numba 用：壓扁 ragged array ──
    # ray_offsets[k] = ray_flat 中第 k 條射線的起始 index
    # ray_lengths[k] = 第 k 條射線的採樣點數
    ray_offsets = np.zeros(N, dtype=np.int32)
    ray_lengths = np.zeros(N, dtype=np.int32)
    total_pts   = sum(len(pts) for pts in ray_samples_list)
    ray_flat    = np.zeros((total_pts, 2), dtype=np.float32)

    idx = 0
    for k, pts in enumerate(ray_samples_list):
        L = len(pts)
        ray_offsets[k] = idx
        ray_lengths[k] = L
        if L > 0:
            ray_flat[idx:idx+L] = pts
        idx += L

    return (fov_rc, fov_rc_np, fov_fwd, fov_right,
            ray_samples_list, ray_flat, ray_offsets, ray_lengths)


(_FOV_RC, _FOV_RC_NP, _FOV_FWD, _FOV_RIGHT,
 _RAY_SAMPLES, _RAY_FLAT, _RAY_OFFSETS, _RAY_LENGTHS) = _precompute_fov()


# ==========================================
# ★ Numba JIT 加速函式（全域，Class 外）
# ==========================================

@njit(cache=True)
def njit_has_line_of_sight(x1, y1, x2, y2, grid_np, tile_size, cols, rows):
    """LOS 檢查：從 (x1,y1) 到 (x2,y2) 是否無牆壁遮擋。"""
    dx   = x2 - x1
    dy   = y2 - y1
    dist = math.sqrt(dx * dx + dy * dy)
    if dist <= 0.0:
        return True
    steps = max(1, int(dist / (tile_size * 0.5)))
    sx    = dx / steps
    sy    = dy / steps
    cx    = x1
    cy    = y1
    for _ in range(steps - 1):
        cx += sx
        cy += sy
        tc = int(cx // tile_size)
        tr = int(cy // tile_size)
        if 0 <= tc < cols and 0 <= tr < rows:
            if grid_np[tr, tc] == 1:
                return False
        else:
            return False
    return True


@njit(cache=True)
def njit_compute_fov(
    ax, ay,
    fwd_x, fwd_y, rgt_x, rgt_y,
    grid_np,
    fov_rc_np,      # (N, 2) int32
    fov_fwd,        # (N,)   float32
    fov_right,      # (N,)   float32
    ray_flat,       # (total_pts, 2) float32  — 壓扁的採樣點
    ray_offsets,    # (N,) int32
    ray_lengths,    # (N,) int32
    tile_size,
    cols,
    rows,
    view_size,
):
    """
    計算 FOV 地形通道 (Ch0)。
    回傳 shape=(view_size, view_size) float32 array。
    初始值 -1.0（迷霧），可見格填入 0.0（空地）或 1.0（牆壁）。
    """
    ch0 = np.full((view_size, view_size), -1.0, dtype=np.float32)
    N   = fov_rc_np.shape[0]

    for k in range(N):
        r_idx = fov_rc_np[k, 0]
        c_idx = fov_rc_np[k, 1]
        ft    = fov_fwd[k]
        rt    = fov_right[k]

        wx = ax + fwd_x * ft * tile_size + rgt_x * rt * tile_size
        wy = ay + fwd_y * ft * tile_size + rgt_y * rt * tile_size

        tc   = int(wx // tile_size)
        tr_v = int(wy // tile_size)

        if not (0 <= tc < cols and 0 <= tr_v < rows):
            ch0[r_idx, c_idx] = 1.0
            continue

        # LOS：沿壓扁陣列中的採樣點步進
        off = ray_offsets[k]
        L   = ray_lengths[k]
        blocked = False
        for j in range(L):
            fft = ray_flat[off + j, 0]
            rrt = ray_flat[off + j, 1]
            ix  = ax + fwd_x * fft * tile_size + rgt_x * rrt * tile_size
            iy  = ay + fwd_y * fft * tile_size + rgt_y * rrt * tile_size
            ic  = int(ix // tile_size)
            ir  = int(iy // tile_size)
            if 0 <= ic < cols and 0 <= ir < rows:
                if grid_np[ir, ic] == 1:
                    blocked = True
                    break
            else:
                blocked = True
                break

        if not blocked:
            ch0[r_idx, c_idx] = float(grid_np[tr_v, tc])

    return ch0


# ── 觸發一次 JIT 編譯（在 import 時完成，訓練時不卡） ──
def _warmup_numba():
    dummy_grid = np.zeros((ROWS, COLS), dtype=np.int8)
    njit_has_line_of_sight(0.0, 0.0, 1.0, 1.0, dummy_grid,
                           float(TILE_SIZE), COLS, ROWS)
    njit_compute_fov(
        0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
        dummy_grid,
        _FOV_RC_NP, _FOV_FWD, _FOV_RIGHT,
        _RAY_FLAT, _RAY_OFFSETS, _RAY_LENGTHS,
        float(TILE_SIZE), COLS, ROWS, VIEW_SIZE,
    )

print("⚡ Numba JIT 預熱中（首次需約 5-10 秒）...")
_warmup_numba()
print("⚡ Numba JIT 預熱完成！")


# ==========================================
# Agent
# ==========================================
class Agent:
    def __init__(self, x, y, color):
        self.x, self.y = x, y
        self.radius    = 15
        self.color     = color
        self.speed     = 3
        self.angle     = 0
        self.hp        = 100
        self.max_hp    = 100
        self.attack_cooldown = 0
        self.ammo        = 5
        self.max_ammo    = 5
        self.reload_timer = 0
        self.reload_delay = 180
        self.dash_timer  = 0
        self.dash_cd     = 0
        self.regen_timer = 0


    def step(self, actions, env):
        if self.hp <= 0:
            return False, 0.0
        up, dn, lt, rt, cw, ccw, atk, focus, dash_btn = actions
        fwd_in   = (1 if up  > 0.5 else 0) - (1 if dn  > 0.5 else 0)
        right_in = (1 if rt  > 0.5 else 0) - (1 if lt  > 0.5 else 0)
        turn_in  = (1 if cw  > 0.5 else 0) - (1 if ccw > 0.5 else 0)
        turn_speed = 1.5 if focus > 0.5 else 8.0

        dash_reward = 0.0
        if self.dash_cd > 0:
            self.dash_cd -= 1
        
        if self.dash_timer > 0:
            self.dash_timer -= 1
            cur_speed = self.speed * 3
        else:
            cur_speed = self.speed

        if dash_btn > 0.5 and self.dash_cd == 0 and self.dash_timer == 0 and self.hp > GameConfig.DASH_COST_HP:
            self.dash_timer = 10
            self.dash_cd = 160
            self.hp -= GameConfig.DASH_COST_HP
            dash_reward = GameConfig.DASH_PENALTY
            cur_speed = self.speed * 3

        rad = math.radians(self.angle)
        fx, fy = math.cos(rad), math.sin(rad)
        rx, ry = math.cos(rad + math.pi/2), math.sin(rad + math.pi/2)
        dx = fx * cur_speed * fwd_in + rx * cur_speed * right_in
        dy = fy * cur_speed * fwd_in + ry * cur_speed * right_in

        # 回血邏輯
        if fwd_in == 0 and right_in == 0:
            self.regen_timer += 1
            if self.regen_timer >= 240:
                self.hp = min(self.max_hp, self.hp + GameConfig.REGEN_AMOUNT)
                self.regen_timer = 0
        else:
            self.regen_timer = 0

        other = env.agents[1] if env.agents[0] is self else env.agents[0]
        nx, ny = self.x + dx, self.y + dy
        new_x = self.x
        new_y = self.y
        if not env.is_wall(nx, self.y) and (other.hp <= 0 or
                math.hypot(nx - other.x, self.y - other.y) >= self.radius + other.radius):
            new_x = nx
        if not env.is_wall(self.x, ny) and (other.hp <= 0 or
                math.hypot(self.x - other.x, ny - other.y) >= self.radius + other.radius):
            new_y = ny
        self.x, self.y = new_x, new_y
        self.angle = (self.angle + turn_speed * turn_in) % 360

        # ── 分離推力：防止兩個 AI 互相卡死 ──
        if other.hp > 0:
            dist = math.hypot(self.x - other.x, self.y - other.y)
            min_dist = self.radius + other.radius
            if dist < min_dist and dist > 0:
                push = (min_dist - dist + 1.0) / 2.0
                push_x = (self.x - other.x) / dist * push
                push_y = (self.y - other.y) / dist * push
                if not env.is_wall(self.x + push_x, self.y):
                    self.x += push_x
                if not env.is_wall(self.x, self.y + push_y):
                    self.y += push_y

        if self.ammo == 0:
            self.reload_timer += 1
            if self.reload_timer >= self.reload_delay:
                self.ammo = self.max_ammo
                self.reload_timer = 0

        if self.attack_cooldown > 0:
            self.attack_cooldown -= 1

        did_shoot = False
        if atk > 0.5 and self.attack_cooldown == 0 and self.ammo > 0:
            rad2 = math.radians(self.angle)
            sx   = self.x + math.cos(rad2) * (self.radius + 5)
            sy   = self.y + math.sin(rad2) * (self.radius + 5)
            env.projectiles.append(Projectile(sx, sy, self.angle, owner=self))
            self.attack_cooldown = 15
            self.ammo -= 1
            did_shoot = True
        return did_shoot, dash_reward

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)
        rad = math.radians(self.angle)
        ex  = self.x + math.cos(rad) * self.radius * 1.5
        ey  = self.y + math.sin(rad) * self.radius * 1.5
        pygame.draw.line(screen, (255, 255, 0), (self.x, self.y), (ex, ey), 3)


class Projectile:
    def __init__(self, x, y, angle, owner):
        self.x, self.y = x, y
        self.angle  = angle
        self.speed  = 18
        self.radius = 4
        self.owner  = owner
        self.life   = 35

    def update(self):
        rad = math.radians(self.angle)
        self.x += math.cos(rad) * self.speed
        self.y += math.sin(rad) * self.speed
        self.life -= 1

    def draw(self, screen):
        pygame.draw.circle(screen, (255, 220, 50), (int(self.x), int(self.y)), self.radius)


# ==========================================
# GameEnv
# ==========================================
class GameEnv:
    def __init__(self, render_mode=False):
        self.render_mode = render_mode
        self.bullet_damage = GameConfig.BULLET_DAMAGE
        self.tie_penalty   = GameConfig.TIE_PENALTY
        self.map_pool      = MAPS
        self.screen        = None
        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("AI 訓練環境")
            self.clock = pygame.time.Clock()
        if self.render_mode and not hasattr(self, 'font'):
            pygame.font.init()
            self.font = pygame.font.SysFont(None, 24)

    def set_config(self, bullet_damage, tie_penalty, use_small_maps=False):
        self.bullet_damage = bullet_damage
        self.tie_penalty   = tie_penalty
        self.map_pool      = TINY_MAPS if use_small_maps else MAPS


    def reset(self):
        self.grid    = random.choice(self.map_pool)
        self.grid_np = np.array(self.grid, dtype=np.int8)
        empty_spots  = [
            (c * TILE_SIZE + TILE_SIZE // 2, r * TILE_SIZE + TILE_SIZE // 2)
            for r in range(ROWS) for c in range(COLS)
            if self.grid[r, c] == 0
        ]
        min_dist = TILE_SIZE * 4 if id(self.grid) in SMALL_MAP_IDS else TILE_SIZE * 6
        for _ in range(200):
            spawns = random.sample(empty_spots, 2)
            if math.hypot(spawns[0][0] - spawns[1][0],
                          spawns[0][1] - spawns[1][1]) > min_dist:
                break
        self.agents = [
            Agent(spawns[0][0], spawns[0][1], (0, 100, 255)),
            Agent(spawns[1][0], spawns[1][1], (255, 60, 60)),
        ]
        self.agents[0].angle = random.randint(0, 359)
        self.agents[1].angle = random.randint(0, 359)
        self.projectiles = []
        self.frame_count = 0
        return self.get_states()

    def is_wall(self, x, y):
        c = int(x // TILE_SIZE)
        r = int(y // TILE_SIZE)
        if 0 <= c < COLS and 0 <= r < ROWS:
            return self.grid[r, c] == 1
        return True

    def has_line_of_sight(self, x1, y1, x2, y2):
        """★ 直接呼叫 Numba JIT 版本。"""
        return njit_has_line_of_sight(
            float(x1), float(y1), float(x2), float(y2),
            self.grid_np, float(TILE_SIZE), COLS, ROWS
        )

    # ==========================================
    # ★ 新版觀測空間（Numba 加速 Ch0，Python 處理 Ch1-3）
    # ==========================================
    def _get_local_view(self, agent_idx):
        """
        Ch0 (地形): 牆壁=1.0, 空地=0.0, 迷霧=-1.0   ← Numba JIT
        Ch1 (雷達): 視線內敵人該格=1.0
        Ch2 (威脅): 敵方子彈=1.0, 我方子彈=-1.0
        Ch3 (槍械): 中心=子彈比例, 中心上格=裝填進度
        """
        agent = self.agents[agent_idx]
        enemy = self.agents[1 - agent_idx]

        rad = math.radians(agent.angle)
        fwd_x, fwd_y = math.cos(rad),             math.sin(rad)
        rgt_x, rgt_y = math.cos(rad + math.pi/2), math.sin(rad + math.pi/2)
        ax, ay = agent.x, agent.y

        view = np.zeros((4, VIEW_SIZE, VIEW_SIZE), dtype=np.float32)

        # ── Ch0：Numba JIT 加速地形 ──
        view[0] = njit_compute_fov(
            float(ax), float(ay),
            float(fwd_x), float(fwd_y), float(rgt_x), float(rgt_y),
            self.grid_np,
            _FOV_RC_NP, _FOV_FWD, _FOV_RIGHT,
            _RAY_FLAT, _RAY_OFFSETS, _RAY_LENGTHS,
            float(TILE_SIZE), COLS, ROWS, VIEW_SIZE,
        )

        # ── Ch1：雷達（敵人位置，無 HP）──
        if enemy.hp > 0:
            dx = enemy.x - ax
            dy = enemy.y - ay
            ft = (dx * fwd_x + dy * fwd_y) / TILE_SIZE
            rt = (dx * rgt_x + dy * rgt_y) / TILE_SIZE
            dt = math.hypot(ft, rt)
            ang_e = math.degrees(math.atan2(rt, ft)) if dt > 0 else 0.0
            if (dt <= VIEW_RANGE and abs(ang_e) <= HALF_FOV
                    and self.has_line_of_sight(ax, ay, enemy.x, enemy.y)):
                r_f = VIEW_CENTER - ft
                c_f = VIEW_CENTER + rt
                r0 = int(math.floor(r_f))
                c0 = int(math.floor(c_f))
                dr = r_f - r0
                dc = c_f - c0
                if 0 <= r0 < VIEW_SIZE and 0 <= c0 < VIEW_SIZE: view[1, r0, c0] += (1.0 - dr) * (1.0 - dc)
                if 0 <= r0 < VIEW_SIZE and 0 <= c0+1 < VIEW_SIZE: view[1, r0, c0+1] += (1.0 - dr) * dc
                if 0 <= r0+1 < VIEW_SIZE and 0 <= c0 < VIEW_SIZE: view[1, r0+1, c0] += dr * (1.0 - dc)
                if 0 <= r0+1 < VIEW_SIZE and 0 <= c0+1 < VIEW_SIZE: view[1, r0+1, c0+1] += dr * dc

        # ── Ch2：威脅（子彈）──
        for p in self.projectiles:
            dx = p.x - ax
            dy = p.y - ay
            ft = (dx * fwd_x + dy * fwd_y) / TILE_SIZE
            rt = (dx * rgt_x + dy * rgt_y) / TILE_SIZE
            dt = math.hypot(ft, rt)
            ang_b = math.degrees(math.atan2(rt, ft)) if dt > 0 else 0.0
            if dt <= VIEW_RANGE and abs(ang_b) <= HALF_FOV:
                val = 1.0 if p.owner is not agent else -1.0
                r_f = VIEW_CENTER - ft
                c_f = VIEW_CENTER + rt
                r0 = int(math.floor(r_f))
                c0 = int(math.floor(c_f))
                dr = r_f - r0
                dc = c_f - c0
                if 0 <= r0 < VIEW_SIZE and 0 <= c0 < VIEW_SIZE: view[2, r0, c0] += val * (1.0 - dr) * (1.0 - dc)
                if 0 <= r0 < VIEW_SIZE and 0 <= c0+1 < VIEW_SIZE: view[2, r0, c0+1] += val * (1.0 - dr) * dc
                if 0 <= r0+1 < VIEW_SIZE and 0 <= c0 < VIEW_SIZE: view[2, r0+1, c0] += val * dr * (1.0 - dc)
                if 0 <= r0+1 < VIEW_SIZE and 0 <= c0+1 < VIEW_SIZE: view[2, r0+1, c0+1] += val * dr * dc

        # ── Ch3：狀態面版 ──
        ammo_ratio   = agent.ammo / agent.max_ammo
        reload_ratio = agent.reload_timer / agent.reload_delay if agent.ammo == 0 else 0.0
        hp_ratio     = agent.hp / agent.max_hp
        dash_ratio   = agent.dash_cd / 160.0
        regen_ratio  = agent.regen_timer / 240.0
        
        view[3, VIEW_CENTER,     VIEW_CENTER] = ammo_ratio
        view[3, VIEW_CENTER - 1, VIEW_CENTER] = reload_ratio
        view[3, VIEW_CENTER + 1, VIEW_CENTER] = hp_ratio
        view[3, VIEW_CENTER,     VIEW_CENTER - 1] = dash_ratio
        view[3, VIEW_CENTER,     VIEW_CENTER + 1] = regen_ratio


        return view

    def get_states(self):
        return [self._get_local_view(0), self._get_local_view(1)]

    def _single_step(self, action1, action2):
        self.frame_count += 1
        rewards = [0.0, 0.0]

        s1, d1 = self.agents[0].step(action1, self)
        s2, d2 = self.agents[1].step(action2, self)

        if s1: rewards[0] -= GameConfig.SHOOT_PENALTY
        if s2: rewards[1] -= GameConfig.SHOOT_PENALTY
        
        rewards[0] += d1
        rewards[1] += d2

        # ── 雷達掃描獎勵 (只要敵人在視野內就給極少量分，引導尋敵) ──
        for i in range(2):
            my = self.agents[i]
            en = self.agents[1 - i]
            if en.hp > 0 and my.hp > 0:
                # 這裡要計算相對坐標，邏輯與 _get_local_view 內的雷達一致
                rad = math.radians(my.angle)
                fx, fy = math.cos(rad), math.sin(rad)
                rx, ry = math.cos(rad + math.pi/2), math.sin(rad + math.pi/2)
                dx, dy = en.x - my.x, en.y - my.y
                ft = (dx * fx + dy * fy) / TILE_SIZE
                rt = (dx * rx + dy * ry) / TILE_SIZE
                dist_t = math.hypot(ft, rt)
                ang_t  = math.degrees(math.atan2(rt, ft)) if dist_t > 0 else 0.0
                if dist_t <= VIEW_RANGE and abs(ang_t) <= HALF_FOV and self.has_line_of_sight(my.x, my.y, en.x, en.y):
                    rewards[i] += GameConfig.RADAR_REWARD


        for p in self.projectiles[:]:
            p.update()
            if self.is_wall(p.x, p.y) or p.life <= 0:
                self.projectiles.remove(p)
                continue
            for i, ag in enumerate(self.agents):
                if p.owner is not ag and ag.hp > 0:
                    if math.hypot(p.x - ag.x, p.y - ag.y) < ag.radius + p.radius:
                        ag.hp -= self.bullet_damage
                        rewards[i]     -= GameConfig.DAMAGE_PENALTY
                        rewards[1 - i] += GameConfig.HIT_REWARD
                        if p in self.projectiles:
                            self.projectiles.remove(p)

        rewards[0] -= GameConfig.SURVIVAL_COST
        rewards[1] -= GameConfig.SURVIVAL_COST

        done = False
        # 這是遊戲結束的總開關：涵蓋「有人血量歸零」或「達到 1200 步」
        if self.agents[0].hp <= 0 or self.agents[1].hp <= 0 or self.frame_count >= MAX_FRAMES:
            done = True
            
            # 情況 A：P1 血量高於 P2 (包含 P2 死亡)
            if self.agents[0].hp > self.agents[1].hp:
                rewards[0] += GameConfig.WIN_REWARD
                rewards[1] -= GameConfig.LOSE_PENALTY
                
            # 情況 B：P2 血量高於 P1 (包含 P1 死亡)
            elif self.agents[1].hp > self.agents[0].hp:
                rewards[1] += GameConfig.WIN_REWARD
                rewards[0] -= GameConfig.LOSE_PENALTY
                
            # 積分後製：只要雙方都還活著，就同步扣除生存懲罰 (鼓勵擊殺而非僅是換血)
            if self.agents[0].hp > 0 and self.agents[1].hp > 0:
                rewards[0] -= self.tie_penalty
                rewards[1] -= self.tie_penalty
        return self.get_states(), rewards, done, {}
    def step(self, action1, action2, frame_skip=1):
        total = [0.0, 0.0]
        done  = False
        states, info = None, {}
        for _ in range(frame_skip):
            if done: break
            states, rew, done, info = self._single_step(action1, action2)
            total[0] += rew[0]
            total[1] += rew[1]
        return states, total, done, info

    def render(self, info=""):
        if not self.render_mode: return
        self.screen.fill((20, 20, 30))
        for r in range(ROWS):
            for c in range(COLS):
                if self.grid[r, c] == 1:
                    pygame.draw.rect(self.screen, (100, 100, 120),
                                     (c * TILE_SIZE, r * TILE_SIZE, TILE_SIZE, TILE_SIZE))
        for p in self.projectiles:
            p.draw(self.screen)
        for a in self.agents:
            if a.hp > 0: a.draw(self.screen)
        for i, a in enumerate(self.agents):
            txt = self.font.render(
                f"P{i+1} HP:{a.hp} Ammo:{a.ammo} Reload:{a.reload_timer} DashCD:{a.dash_cd} Regen:{a.regen_timer}", True, (255,255,255))
            self.screen.blit(txt, (10, 10 + i * 20))
        if info:
            self.screen.blit(self.font.render(info, True, (255,220,50)), (10, HEIGHT-30))
        pygame.display.flip()
        self.clock.tick(FPS)

import pygame
import math
import random
import numpy as np

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
FOV_DEGREES = 100.0
HALF_FOV    = FOV_DEGREES / 2.0
VIEW_RANGE  = 14               # 格數

NUM_ACTIONS = 8                # 含第 8 個 Focus 動作

# ==========================================
# 地圖 (0=空地, 1=牆壁)  32×24
# ==========================================
MAP_1 = [
    [1]*32,
    [1]+[0]*30+[1],
    [1]+[0]*30+[1],
    [1,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,1],
    [1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1]+[0]*30+[1],
    [1]+[0]*30+[1],
    [1]+[0]*30+[1],
    [1]+[0]*30+[1],
    [1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1],
    [1,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,1],
    [1]+[0]*30+[1],
    [1]+[0]*30+[1],
    [1]+[0]*30+[1],
    [1]*32,
]
MAP_2 = [
    [1]*32,
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,1,1,1,0,0,1,1,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,1,0,0,1,1,1,0,1],
    [1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1],
    [1,0,1,0,0,0,0,0,0,1,0,0,1,1,1,0,0,1,1,1,0,0,0,0,0,0,0,0,0,1,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,1,1,1,1,1,1,1,1,0,0,1,0,0,0,0,0,0,1,0,0,1,1,1,1,1,1,1,1,0,1],
    [1]+[0]*30+[1],
    [1]+[0]*30+[1],
    [1]+[0]*30+[1],
    [1]+[0]*30+[1],
    [1,0,1,1,1,1,1,1,1,1,0,0,1,0,0,0,0,0,0,1,0,0,1,1,1,1,1,1,1,1,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,1,0,0,0,0,0,0,1,0,0,1,1,1,0,0,1,1,1,0,0,0,0,0,0,0,0,0,1,0,1],
    [1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1],
    [1,0,1,1,1,0,0,1,1,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,1,0,0,1,1,1,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1]+[0]*30+[1],
    [1]+[0]*30+[1],
    [1]+[0]*30+[1],
    [1]+[0]*30+[1],
    [1]*32,
]
MAP_3 = [
    [1]*32,
    [1]+[0]*30+[1],
    [1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,1,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,1],
    [1,0,0,1,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1],
    [1,0,0,0,0,0,0,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,0,0,1],
    [1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1],
    [1]+[0]*30+[1],
    [1]+[0]*30+[1],
    [1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1],
    [1,0,0,1,1,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1],
    [1]+[0]*30+[1],
    [1]+[0]*30+[1],
    [1]+[0]*30+[1],
    [1]+[0]*30+[1],
    [1]+[0]*30+[1],
    [1]*32,
]
MAP_4 = [
    [1]*32,
    [1]+[0]*30+[1],
    [1,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,1],
    [1,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,1],
    [1]+[0]*30+[1],
    [1]+[0]*30+[1],
    [1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1],
    [1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1],
    [1]+[0]*30+[1],
    [1,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,1],
    [1]+[0]*30+[1],
    [1]+[0]*30+[1],
    [1,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,1],
    [1]+[0]*30+[1],
    [1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1],
    [1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1],
    [1]+[0]*30+[1],
    [1,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,1],
    [1,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,1],
    [1]+[0]*30+[1],
    [1]*32,
]
MAP_5 = [
    [1]*32,
    [1]+[0]*30+[1],
    [1,0,1,1,1,1,1,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,1,1,1,1,1,1,0,0,1],
    [1,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,1],
    [1,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,1],
    [1,0,1,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,1,1,0,0,1],
    [1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1],
    [1,0,0,0,0,1,0,0,0,0,1,1,1,0,0,0,0,1,1,1,0,0,0,0,1,0,0,0,0,0,0,1],
    [1]+[0]*30+[1],
    [1]+[0]*30+[1],
    [1]+[0]*30+[1],
    [1]+[0]*30+[1],
    [1,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1],
    [1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1],
    [1,0,1,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,1,1,0,0,1],
    [1,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,1],
    [1,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,1],
    [1,0,1,1,1,1,1,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,1,1,1,1,1,1,0,0,1],
    [1]+[0]*30+[1],
    [1]+[0]*30+[1],
    [1]+[0]*30+[1],
    [1]+[0]*30+[1],
    [1]+[0]*30+[1],
    [1]*32,
]

MAPS = [MAP_1, MAP_2, MAP_3, MAP_4, MAP_5]


# ==========================================
# FOV 視野快取（全域，程式啟動時只算一次）
# ==========================================
def _precompute_fov():
    """
    預計算 15×15 FOV grid 的：
    - in_fov 遮罩
    - 每個可見格子的 (fwd_tile, right_tile) 偏移
    - 每個可見格子的射線中間採樣點列表（用於 LOS 查詢）
    """
    rows_idx = np.arange(VIEW_SIZE)
    cols_idx = np.arange(VIEW_SIZE)
    col_g, row_g = np.meshgrid(cols_idx, rows_idx)
    fwd   = (VIEW_CENTER - row_g).astype(np.float32)
    right = (col_g - VIEW_CENTER).astype(np.float32)
    dist  = np.hypot(fwd, right)
    angle = np.degrees(np.arctan2(right, fwd))
    in_fov = (dist <= VIEW_RANGE) & ((dist == 0) | (np.abs(angle) <= HALF_FOV))

    fov_rc = list(zip(*np.where(in_fov)))   # list of (row, col)

    # 對每個 in_fov 格子，預計算射線採樣點的 (fwd_frac, right_frac)（tile 單位）
    # 從 origin 到 target 線性內插，步長 ≈ 0.4 tile，取中間點（不含端點）
    ray_samples = []   # parallel list with fov_rc
    for (r, c) in fov_rc:
        ft  = float(fwd[r, c])
        rt  = float(right[r, c])
        d   = float(dist[r, c])
        if d <= 0:
            ray_samples.append(np.empty((0, 2), dtype=np.float32))
        else:
            n   = max(2, int(d / 0.4) + 2)
            ts  = np.linspace(0.0, 1.0, n)[1:-1]          # 中間點
            pts = np.column_stack([ft * ts, rt * ts]).astype(np.float32)
            ray_samples.append(pts)

    fov_fwd   = np.array([fwd[r, c]   for r, c in fov_rc], dtype=np.float32)
    fov_right = np.array([right[r, c] for r, c in fov_rc], dtype=np.float32)

    return fov_rc, fov_fwd, fov_right, ray_samples

_FOV_RC, _FOV_FWD, _FOV_RIGHT, _RAY_SAMPLES = _precompute_fov()


class Agent:
    def __init__(self, x, y, color):
        self.x, self.y = x, y
        self.radius    = 15
        self.color     = color
        self.speed     = 3
        self.angle     = 0
        self.hp        = 40
        self.max_hp    = 40
        self.attack_cooldown   = 0
        self.pos_history       = []
        self.stall_frame_counter = 0
        self.ammo        = 5
        self.max_ammo    = 5
        self.reload_timer = 0
        self.reload_delay = 60

    def step(self, actions, env):
        # actions = [上, 下, 左, 右, 順時針, 逆時針, 攻擊, Focus]
        if self.hp <= 0:
            return False, 0.0

        up, dn, lt, rt, cw, ccw, atk, focus = actions

        fwd_in   = (1 if up   > 0.5 else 0) - (1 if dn  > 0.5 else 0)
        right_in = (1 if rt   > 0.5 else 0) - (1 if lt  > 0.5 else 0)
        turn_in  = (1 if cw   > 0.5 else 0) - (1 if ccw > 0.5 else 0)

        turn_speed = 1.5 if focus > 0.5 else 8.0

        rad = math.radians(self.angle)
        fx, fy = math.cos(rad), math.sin(rad)
        rx, ry = math.cos(rad + math.pi/2), math.sin(rad + math.pi/2)

        dx = fx * self.speed * fwd_in + rx * self.speed * right_in
        dy = fy * self.speed * fwd_in + ry * self.speed * right_in

        other = env.agents[1] if env.agents[0] is self else env.agents[0]

        nx, ny = self.x + dx, self.y + dy
        if not env.is_wall(nx, self.y) and (other.hp <= 0 or
                math.hypot(nx - other.x, self.y - other.y) >= self.radius + other.radius):
            self.x = nx
        if not env.is_wall(self.x, ny) and (other.hp <= 0 or
                math.hypot(self.x - other.x, ny - other.y) >= self.radius + other.radius):
            self.y = ny

        self.angle = (self.angle + turn_speed * turn_in) % 360

        if self.ammo < self.max_ammo:
            self.reload_timer += 1
            if self.reload_timer >= self.reload_delay:
                self.ammo += 1
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
            did_shoot  = True

        # 防發呆
        stall_window    = 36
        stall_threshold = 30.0
        self.pos_history.append((self.x, self.y))
        self.stall_frame_counter += 1
        stall_penalty = 0.0
        if self.stall_frame_counter >= stall_window:
            ox, oy = self.pos_history[0]
            disp = math.hypot(self.x - ox, self.y - oy)
            if disp < stall_threshold:
                stall_penalty = -0.5 * (1.0 - disp / stall_threshold)
            self.pos_history.clear()
            self.stall_frame_counter = 0

        return did_shoot, stall_penalty

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
        self.life   = 60

    def update(self):
        rad = math.radians(self.angle)
        self.x += math.cos(rad) * self.speed
        self.y += math.sin(rad) * self.speed
        self.life -= 1

    def draw(self, screen):
        pygame.draw.circle(screen, (255, 220, 50), (int(self.x), int(self.y)), self.radius)


class GameEnv:
    def __init__(self, render_mode=False):
        self.render_mode = render_mode
        self.screen = None
        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("AI 訓練環境")
            self.clock = pygame.time.Clock()
            self.font  = pygame.font.SysFont(None, 24)
        self.reset()

    def reset(self):
        self.grid    = random.choice(MAPS)
        self.grid_np = np.array(self.grid, dtype=np.int8)  # NumPy 版加速索引

        empty_spots = [
            (c * TILE_SIZE + TILE_SIZE // 2, r * TILE_SIZE + TILE_SIZE // 2)
            for r in range(ROWS) for c in range(COLS)
            if self.grid[r][c] == 0
        ]

        # 確保兩者相距夠遠
        for _ in range(200):
            spawns = random.sample(empty_spots, 2)
            if math.hypot(spawns[0][0] - spawns[1][0],
                          spawns[0][1] - spawns[1][1]) > TILE_SIZE * 6:
                break

        self.agents = [
            Agent(spawns[0][0], spawns[0][1], (0, 100, 255)),
            Agent(spawns[1][0], spawns[1][1], (255, 60, 60)),
        ]
        self.agents[0].angle = random.randint(0, 359)
        self.agents[1].angle = random.randint(0, 359)
        self.projectiles   = []
        self.frame_count   = 0
        return self.get_states()

    def is_wall(self, x, y):
        c = int(x // TILE_SIZE)
        r = int(y // TILE_SIZE)
        if 0 <= c < COLS and 0 <= r < ROWS:
            return self.grid[r][c] == 1
        return True

    def has_line_of_sight(self, x1, y1, x2, y2):
        dist = math.hypot(x2 - x1, y2 - y1)
        if dist <= 0:
            return True
        steps = max(1, int(dist / (TILE_SIZE / 2)))
        dx = (x2 - x1) / steps
        dy = (y2 - y1) / steps
        cx, cy = x1, y1
        for _ in range(steps - 1):
            cx += dx
            cy += dy
            if self.is_wall(cx, cy):
                return False
        return True

    # ==========================================
    # ★ 視野式局部地圖 (FOV 15×15，全快取射線)
    # ==========================================
    def _get_local_view(self, agent_idx):
        """
        Ch0: 地形 (1=牆, 0=空地, -1=迷霧)
        Ch1: 自身 (hp/max_hp 於中心格)
        Ch2: 敵人位置 (hp/max_hp，若可見)
        Ch3: 子彈 (+1我方, -1敵方)
        """
        agent  = self.agents[agent_idx]
        enemy  = self.agents[1 - agent_idx]
        grid   = self.grid_np

        view = np.full((4, VIEW_SIZE, VIEW_SIZE), -1.0, dtype=np.float32)
        view[1:] = 0.0

        rad   = math.radians(agent.angle)
        fwd_x, fwd_y   = math.cos(rad),              math.sin(rad)
        rgt_x, rgt_y   = math.cos(rad + math.pi/2),  math.sin(rad + math.pi/2)
        ax, ay = agent.x, agent.y

        # ── 地形 + LOS ──
        for k, (r_idx, c_idx) in enumerate(_FOV_RC):
            ft = float(_FOV_FWD[k])
            rt = float(_FOV_RIGHT[k])

            # 目標格的世界中心
            wx = ax + fwd_x * ft * TILE_SIZE + rgt_x * rt * TILE_SIZE
            wy = ay + fwd_y * ft * TILE_SIZE + rgt_y * rt * TILE_SIZE

            tc = int(wx // TILE_SIZE)
            tr_v = int(wy // TILE_SIZE)

            if not (0 <= tc < COLS and 0 <= tr_v < ROWS):
                view[0, r_idx, c_idx] = 1.0
                continue

            # LOS：沿射線採樣點查詢 grid_np
            pts = _RAY_SAMPLES[k]      # (N, 2)  [fwd_frac, right_frac]
            blocked = False
            for i in range(len(pts)):
                fft, rrt = pts[i, 0], pts[i, 1]
                ix = ax + fwd_x * fft * TILE_SIZE + rgt_x * rrt * TILE_SIZE
                iy = ay + fwd_y * fft * TILE_SIZE + rgt_y * rrt * TILE_SIZE
                ic = int(ix // TILE_SIZE)
                ir = int(iy // TILE_SIZE)
                if 0 <= ic < COLS and 0 <= ir < ROWS:
                    if grid[ir, ic] == 1:
                        blocked = True
                        break
                else:
                    blocked = True
                    break

            if not blocked:
                view[0, r_idx, c_idx] = float(grid[tr_v, tc])

        # ── 自身 ──
        view[1, VIEW_CENTER, VIEW_CENTER] = agent.hp / agent.max_hp

        # ── 敵人 ──
        if enemy.hp > 0:
            dx = enemy.x - ax
            dy = enemy.y - ay
            ft = (dx * fwd_x + dy * fwd_y) / TILE_SIZE
            rt = (dx * rgt_x + dy * rgt_y) / TILE_SIZE
            dt = math.hypot(ft, rt)
            er = VIEW_CENTER - round(ft)
            ec = VIEW_CENTER + round(rt)
            ang_e = math.degrees(math.atan2(rt, ft)) if dt > 0 else 0.0
            if (0 <= er < VIEW_SIZE and 0 <= ec < VIEW_SIZE
                    and dt <= VIEW_RANGE and abs(ang_e) <= HALF_FOV
                    and self.has_line_of_sight(ax, ay, enemy.x, enemy.y)):
                view[2, er, ec] = enemy.hp / enemy.max_hp

        # ── 子彈 ──
        for p in self.projectiles:
            dx = p.x - ax
            dy = p.y - ay
            ft = (dx * fwd_x + dy * fwd_y) / TILE_SIZE
            rt = (dx * rgt_x + dy * rgt_y) / TILE_SIZE
            dt = math.hypot(ft, rt)
            br = VIEW_CENTER - round(ft)
            bc = VIEW_CENTER + round(rt)
            ang_b = math.degrees(math.atan2(rt, ft)) if dt > 0 else 0.0
            if (0 <= br < VIEW_SIZE and 0 <= bc < VIEW_SIZE
                    and dt <= VIEW_RANGE and abs(ang_b) <= HALF_FOV):
                view[3, br, bc] = 1.0 if p.owner is agent else -1.0

        return view

    def get_states(self):
        return [self._get_local_view(0), self._get_local_view(1)]

    # ==========================================
    # 遊戲步進
    # ==========================================
    def _single_step(self, action1, action2):
        self.frame_count += 1
        rewards = [0.0, 0.0]

        d_before = math.hypot(self.agents[0].x - self.agents[1].x,
                               self.agents[0].y - self.agents[1].y)

        s1, p1 = self.agents[0].step(action1, self)
        s2, p2 = self.agents[1].step(action2, self)

        d_after = math.hypot(self.agents[0].x - self.agents[1].x,
                              self.agents[0].y - self.agents[1].y)
        dd = d_before - d_after

        rewards[0] += dd * 0.02 + p1
        rewards[1] += dd * 0.02 + p2

        if s1: rewards[0] -= 0.01
        if s2: rewards[1] -= 0.01

        # 瞄準獎勵
        for i in range(2):
            my = self.agents[i]
            en = self.agents[1 - i]
            dx = en.x - my.x
            dy = en.y - my.y
            ta = math.degrees(math.atan2(dy, dx)) % 360
            df = (ta - my.angle) % 360
            if df > 180: df = 360 - df
            if self.has_line_of_sight(my.x, my.y, en.x, en.y) and df < 90.0:
                aq = (90.0 - df) / 90.0
                db = max(0.5, min(1.5, 200.0 / max(math.hypot(dx, dy), 1.0)))
                rewards[i] += 0.08 * aq * db

        # 子彈
        for p in self.projectiles[:]:
            p.update()
            if self.is_wall(p.x, p.y) or p.life <= 0:
                self.projectiles.remove(p)
                continue
            for i, ag in enumerate(self.agents):
                if p.owner is not ag and ag.hp > 0:
                    if math.hypot(p.x - ag.x, p.y - ag.y) < ag.radius + p.radius:
                        ag.hp -= 20
                        rewards[i]     -= 2.0
                        rewards[1 - i] += 2.0
                        if p in self.projectiles:
                            self.projectiles.remove(p)

        rewards[0] -= 0.01
        rewards[1] -= 0.01

        done = False
        if self.agents[0].hp <= 0 or self.agents[1].hp <= 0 or self.frame_count >= 400:
            done = True
            if self.agents[0].hp > self.agents[1].hp:
                rewards[0] += 100.0; rewards[1] -= 30.0
            elif self.agents[1].hp > self.agents[0].hp:
                rewards[1] += 100.0; rewards[0] -= 30.0

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
                if self.grid[r][c] == 1:
                    pygame.draw.rect(self.screen, (100, 100, 120),
                                     (c * TILE_SIZE, r * TILE_SIZE, TILE_SIZE, TILE_SIZE))
        for p in self.projectiles:
            p.draw(self.screen)
        for a in self.agents:
            if a.hp > 0: a.draw(self.screen)
        for i, a in enumerate(self.agents):
            txt = self.font.render(f"P{i+1} HP:{a.hp} Ammo:{a.ammo}", True, (255,255,255))
            self.screen.blit(txt, (10, 10 + i * 20))
        if info:
            self.screen.blit(self.font.render(info, True, (255,220,50)), (10, HEIGHT-30))
        pygame.display.flip()
        self.clock.tick(FPS)

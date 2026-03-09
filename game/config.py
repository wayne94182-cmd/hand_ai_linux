import math
from dataclasses import dataclass

TILE_SIZE = 40
COLS, ROWS = 32, 24
WIDTH, HEIGHT = COLS * TILE_SIZE, ROWS * TILE_SIZE
FPS = 60

VIEW_SIZE = 15
VIEW_CENTER = 4
FOV_DEGREES = 130.0
HALF_FOV = FOV_DEGREES / 2.0
VIEW_RANGE = 10

NUM_ACTIONS = 9
MAX_FRAMES = 1200


class GameConfig:
    SHOOT_PENALTY = 0.04
    RADAR_REWARD = 0.005
    SURVIVAL_COST = 0.01
    NPC_SURVIVAL_COST = 0.003
    DASH_COST_HP = 5
    DASH_PENALTY = -0.2
    REGEN_AMOUNT = 20

    AGENT_BASE_SPEED = 3.0
    AGENT_MAX_HP = 100
    INDIVIDUAL_REWARD_WEIGHT = 0.6
    TEAM_REWARD_WEIGHT = 0.4

    HIT_REWARD_COEF = 0.05
    DOWN_REWARD = 10.0
    DAMAGE_PENALTY_COEF = 0.05
    WIN_REWARD = 10.0
    LOSE_PENALTY = 5.0

    BULLET_DAMAGE = 20
    TIE_PENALTY = 6.0
    COLLISION_PENALTY = 0.005

    aim_reward = 0.00
    NPC_KILL_REWARD = 10.0
    ALIVE_NPC_PENALTY = 5.0
    BE_REVIVED_REWARD = 3.0
    REVIVE_REWARD = 5.0
    BE_DOWNED_PENALTY = 5.0


@dataclass
class AudioConfig:
    FOOTSTEP_EXPAND_SPEED: float = 90.0
    FOOTSTEP_MAX_RADIUS:   float = 240.0
    RELOAD_MAX_RADIUS:     float = 360.0
    GUNSHOT_MAX_RADIUS:    float = 600.0
    EXPLOSION_MAX_RADIUS:  float = 960.0


@dataclass
class NpcConfig:
    AVOID_RADIUS:      float = 150.0
    RETREAT_TIMER_SEC: int   = 4


@dataclass(frozen=True)
class StageSpec:
    stage_id: int
    name: str
    mode: str
    enemy_hp: int
    enemy_damage: int
    bullet_damage: int
    enemy_fire_rate: float
    enemy_spread_deg: float
    enemy_can_shoot: bool
    enemy_mobile: bool
    teammate_count: int
    enemy_count: int
    max_frames: int = MAX_FRAMES
    # 動作誤差（比例）：0.05 = ±5%，對移動速度與轉向速度各乘上 uniform(1-pct, 1+pct) 的隨機縮放
    move_noise_pct: float = 0.0      # 移動速度誤差比例，例如 0.05 = ±5%
    rotation_noise_pct: float = 0.0  # 轉向速度誤差比例，例如 0.05 = ±5%
    infinite_ammo: bool = False          # Stage 0 專用：無限彈匣
    body_speed_range: tuple = (1.0, 1.0) # log-uniform 身體速度倍率範圍
    body_rot_range:   tuple = (1.0, 1.0) # log-uniform 轉向速度倍率範圍
    has_poison_zone:  bool  = False       # 是否啟用毒圈
    map_pool_key:     str   = "small"    # "small" / "medium" / "large"
    n_learning_agents: int  = 1          # 本階段的 learning agent 數量


STAGE_SPECS = {

    # Stage 0：瞄準期｜單人、無限彈匣、靜止木樁、小地圖
    0: StageSpec(
        stage_id=0, name="瞄準期", mode="scripted",
        enemy_hp=100, enemy_damage=0, bullet_damage=50,
        enemy_fire_rate=0.0, enemy_spread_deg=0.0,
        enemy_can_shoot=False, enemy_mobile=False,
        teammate_count=0, enemy_count=3,
        max_frames=900,
        move_noise_pct=0.02, rotation_noise_pct=0.02,
        infinite_ammo=True,
        body_speed_range=(0.95, 1.05), body_rot_range=(0.95, 1.05),
        has_poison_zone=False, map_pool_key="small",
        n_learning_agents=1,
    ),

    # Stage 1：打靶期｜雙人、開局有武器需撿彈匣、木樁、小地圖
    1: StageSpec(
        stage_id=1, name="打靶期", mode="scripted",
        enemy_hp=100, enemy_damage=0, bullet_damage=35,
        enemy_fire_rate=0.0, enemy_spread_deg=0.0,
        enemy_can_shoot=False, enemy_mobile=False,
        teammate_count=0, enemy_count=3,
        max_frames=1200,
        move_noise_pct=0.05, rotation_noise_pct=0.05,
        infinite_ammo=False,
        body_speed_range=(0.90, 1.10), body_rot_range=(0.90, 1.10),
        has_poison_zone=False, map_pool_key="small+medium",
        n_learning_agents=2,
    ),

    # Stage 2：追獵期｜雙人、逃跑NPC、需預判瞄準、中型地圖
    2: StageSpec(
        stage_id=2, name="追獵期", mode="scripted",
        enemy_hp=100, enemy_damage=0, bullet_damage=25,
        enemy_fire_rate=0.0, enemy_spread_deg=0.0,
        enemy_can_shoot=False, enemy_mobile=True,
        teammate_count=0, enemy_count=3,
        max_frames=1500,
        move_noise_pct=0.05, rotation_noise_pct=0.05,
        infinite_ammo=False,
        body_speed_range=(0.90, 1.10), body_rot_range=(0.90, 1.10),
        has_poison_zone=False, map_pool_key="medium",
        n_learning_agents=2,
    ),

    # Stage 3：生存期｜雙人、NPC會開槍追擊、學閃避脫戰與治療、中型地圖
    3: StageSpec(
        stage_id=3, name="生存期", mode="scripted",
        enemy_hp=100, enemy_damage=20, bullet_damage=20,
        enemy_fire_rate=2.0, enemy_spread_deg=22.5,
        enemy_can_shoot=True, enemy_mobile=True,
        teammate_count=0, enemy_count=3,
        max_frames=1500,
        move_noise_pct=0.08, rotation_noise_pct=0.08,
        infinite_ammo=False,
        body_speed_range=(0.85, 1.15), body_rot_range=(0.85, 1.15),
        has_poison_zone=False, map_pool_key="medium",
        n_learning_agents=2,
    ),

    # Stage 4：戰術期｜三人、大地圖、毒圈、強NPC
    4: StageSpec(
        stage_id=4, name="戰術期", mode="scripted",
        enemy_hp=150, enemy_damage=20, bullet_damage=20,
        enemy_fire_rate=4.0, enemy_spread_deg=20.0,
        enemy_can_shoot=True, enemy_mobile=True,
        teammate_count=0, enemy_count=4,
        max_frames=5400,
        move_noise_pct=0.10, rotation_noise_pct=0.10,
        infinite_ammo=False,
        body_speed_range=(0.85, 1.15), body_rot_range=(0.85, 1.15),
        has_poison_zone=True, map_pool_key="large+medium",
        n_learning_agents=3,
    ),

    # Stage 5：自我博弈｜三人兩隊、大地圖、毒圈
    5: StageSpec(
        stage_id=5, name="自我博弈", mode="self_play",
        enemy_hp=100, enemy_damage=20, bullet_damage=20,
        enemy_fire_rate=0.0, enemy_spread_deg=0.0,
        enemy_can_shoot=True, enemy_mobile=True,
        teammate_count=0, enemy_count=3,
        max_frames=5400,
        move_noise_pct=0.12, rotation_noise_pct=0.12,
        infinite_ammo=False,
        body_speed_range=(0.80, 1.25), body_rot_range=(0.80, 1.25),
        has_poison_zone=True, map_pool_key="large",
        n_learning_agents=3,
    ),

    # Stage 6：多隊博弈｜三人、30%名人堂舊AI參戰、大地圖、毒圈
    6: StageSpec(
        stage_id=6, name="多隊博弈", mode="hall_of_fame",
        enemy_hp=120, enemy_damage=20, bullet_damage=20,
        enemy_fire_rate=3.0, enemy_spread_deg=15.0,
        enemy_can_shoot=True, enemy_mobile=True,
        teammate_count=0, enemy_count=3,
        max_frames=5400,
        move_noise_pct=0.15, rotation_noise_pct=0.15,
        infinite_ammo=False,
        body_speed_range=(0.80, 1.25), body_rot_range=(0.80, 1.25),
        has_poison_zone=True, map_pool_key="large",
        n_learning_agents=3,
    ),
}


def get_stage_spec(stage_id: int) -> StageSpec:
    if stage_id not in STAGE_SPECS:
        raise ValueError(f"Unknown stage id: {stage_id}")
    return STAGE_SPECS[stage_id]

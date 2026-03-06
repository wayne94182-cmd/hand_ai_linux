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
    SHOOT_PENALTY = 0.02
    RADAR_REWARD = 0.005
    SURVIVAL_COST = 0.01
    NPC_SURVIVAL_COST = 0.003
    DASH_COST_HP = 5
    DASH_PENALTY = -0.2
    REGEN_AMOUNT = 20

    HIT_REWARD = 1.0
    DAMAGE_PENALTY = 1.0
    WIN_REWARD = 10.0
    LOSE_PENALTY = 5.0

    BULLET_DAMAGE = 20
    TIE_PENALTY = 6.0
    COLLISION_PENALTY = 0.005

    aim_reward = 0.00
    NPC_KILL_REWARD = 10.0
    ALIVE_NPC_PENALTY = 5.0


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


STAGE_SPECS = {
    # 參數順序:
    # (階段ID, 階段名稱, 模式, 敵人血量, 敵人子彈傷害, AI子彈傷害, 敵人每秒射速, 敵人射擊偏移角度,
    #  敵人會不會開槍, 敵人會不會移動, 隊友數量, 敵人數量, 最大幀數, 移動速度誤差%, 轉向速度誤差%)
    # 誤差比例說明：0.05 = ±5%，0.10 = ±10%，0.0 = 無誤差
    0: StageSpec(0, "基礎期",  "scripted",  100, 20, 50, 0.0, 0.0,  False, False, 0, 3, MAX_FRAMES, move_noise_pct=0.0,  rotation_noise_pct=0.0),
    # S0: 基礎木樁模式，沒有隊友。讓AI專心尋找敵人與射擊。無動作誤差。
    1: StageSpec(1, "打靶期",  "scripted",  100, 20, 50, 0.0, 0.0,  False, False, 1, 3, MAX_FRAMES, move_noise_pct=0.0,  rotation_noise_pct=0.0),
    # S1: 木樁模式。敵人不會開槍也不會移動。無動作誤差。
    2: StageSpec(2, "追獵期",  "scripted",  100, 20, 35, 0.0, 0.0,  False, True,  1, 3, MAX_FRAMES, move_noise_pct=0.05,  rotation_noise_pct=0.05),
    # S2: 跑酷模式。敵人會逃跑，不會開槍。無動作誤差。
    3: StageSpec(3, "生存期",  "scripted",  100, 20, 20, 2.0, 22.5, True,  True,  1, 3, MAX_FRAMES, move_noise_pct=0.05, rotation_noise_pct=0.05),
    # S3: 砲台模式。敵人開槍。輕微動作誤差 (移動±5%, 轉向±5%)。
    4: StageSpec(4, "戰術期",  "scripted",  150, 20, 20, 5.0, 22.5, True,  True,  1, 3, MAX_FRAMES, move_noise_pct=0.10, rotation_noise_pct=0.10),
    # S4: 菁英怪模式。敵人狂射。中等動作誤差 (移動±10%, 轉向±10%)。
    5: StageSpec(5, "自我博弈", "self_play", 100, 20, 20, 0.0, 0.0,  True,  True,  0, 1, MAX_FRAMES, move_noise_pct=0.10, rotation_noise_pct=0.10),
    # S5: AI對戰。動作誤差同 S4 (±10%)。
    6: StageSpec(6, "團隊期",  "team_2v2",  150, 20, 20, 3.0, 65.0, True,  True,  1, 2, MAX_FRAMES, move_noise_pct=0.15, rotation_noise_pct=0.15),
    # S6: 2v2 模式。最大動作誤差 (移動±15%, 轉向±15%)。
}


def get_stage_spec(stage_id: int) -> StageSpec:
    if stage_id not in STAGE_SPECS:
        raise ValueError(f"Unknown stage id: {stage_id}")
    return STAGE_SPECS[stage_id]

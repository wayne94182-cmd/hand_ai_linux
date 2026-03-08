"""
game/npc.py — NPC 行為模組
從 GameEnv 搬移出來的所有 NPC / Bot AI 邏輯，
改為接受 env 作為第一個參數的函式。
"""
import math
import random

from game.config import (
    TILE_SIZE, FPS, NUM_ACTIONS,
    VIEW_RANGE, HALF_FOV,
    GameConfig, NpcConfig,
)


# ═══════════════════════════════════════════════════════
#  基礎行為
# ═══════════════════════════════════════════════════════

def random_wander_actions(env, agent, allow_shoot=False):
    if agent.bot_turn_hold <= 0:
        agent.bot_turn_hold = random.randint(15, 80)
        agent._turn_dir = random.choice([-1, 0, 1])
    agent.bot_turn_hold -= 1

    actions = [0.0] * NUM_ACTIONS
    actions[0] = 1.0
    turn_dir = getattr(agent, "_turn_dir", 0)
    if turn_dir > 0:
        actions[4] = 1.0
    elif turn_dir < 0:
        actions[5] = 1.0
    if allow_shoot and random.random() < 0.03:
        actions[6] = 1.0
    return actions


def aim_and_shoot_actions(env, shooter, target, fire_rate, spread_deg, use_fov=True):
    actions = [0.0] * NUM_ACTIONS
    if not target.alive() or not shooter.alive():
        return actions

    dx = target.x - shooter.x
    dy = target.y - shooter.y
    desired = math.degrees(math.atan2(dy, dx)) % 360

    diff = (desired - shooter.angle + 540) % 360 - 180
    abs_diff = abs(diff)

    if abs_diff > 45:
        if diff > 0:
            actions[4] = 1.0
        else:
            actions[5] = 1.0

    allowed_angle = min(HALF_FOV, 45.0) if use_fov else 45.0
    angle_ok = abs_diff <= allowed_angle
    dist_tiles = math.hypot(dx, dy) / TILE_SIZE
    if fire_rate > 0 and dist_tiles <= VIEW_RANGE and angle_ok and env.has_line_of_sight(shooter.x, shooter.y, target.x, target.y):
        jitter = random.uniform(0.0, spread_deg)
        signed = jitter if random.random() < 0.5 else -jitter
        shooter.angle = (desired + signed) % 360
        actions[6] = 1.0
    return actions


def avoidance_actions(env, agent, avoid_radius=NpcConfig.AVOID_RADIUS):
    rx, ry = 0.0, 0.0
    count = 0
    for other in env.all_agents:
        if other is agent or not other.alive():
            continue
        dx = agent.x - other.x
        dy = agent.y - other.y
        dist = math.hypot(dx, dy)
        if 0 < dist < avoid_radius:
            force = 1.0 / dist
            rx += dx / dist * force
            ry += dy / dist * force
            count += 1

    if count == 0:
        return None

    desired = math.degrees(math.atan2(ry, rx)) % 360
    diff = (desired - agent.angle + 540) % 360 - 180

    actions = [0.0] * NUM_ACTIONS
    actions[0] = 1.0  # Forward

    if abs(diff) > 15:
        if diff > 0:
            actions[4] = 1.0
        else:
            actions[5] = 1.0
    return actions


# ═══════════════════════════════════════════════════════
#  逃跑行為（Stage 2）
# ═══════════════════════════════════════════════════════

def flee_actions(env, agent, target):
    actions = [0.0] * NUM_ACTIONS

    has_los = env.has_line_of_sight(agent.x, agent.y, target.x, target.y)
    dist = math.hypot(agent.x - target.x, agent.y - target.y)

    # 初始化逃跑記憶屬性
    if not hasattr(agent, 'flee_memory_timer'):
        agent.flee_memory_timer = 0
        agent.last_threat_x = target.x
        agent.last_threat_y = target.y
    if not hasattr(agent, 'flee_zigzag_timer'):
        agent.flee_zigzag_timer = 0
        agent.flee_angle_offset = random.choice([-50, -30, 30, 50])

    # 只有被看到且在危險距離內才觸發逃跑（無 LOS 就不驚覺）
    if has_los and dist < 350:
        agent.flee_memory_timer = 90  # 刷新逃跑慣性，約 1.5 秒
        agent.last_threat_x = target.x
        agent.last_threat_y = target.y
    else:
        if agent.flee_memory_timer > 0:
            agent.flee_memory_timer -= 1  # 仍在驚嚇狀態中
        else:
            return random_wander_actions(env, agent, allow_shoot=False)

    # 計算反向逃跑向量（基於最後記憶的威脅位置）
    dx = agent.x - agent.last_threat_x
    dy = agent.y - agent.last_threat_y
    base_flee_angle = math.degrees(math.atan2(dy, dx)) % 360

    # 蛇行走位 Z 字型
    if agent.flee_zigzag_timer <= 0:
        agent.flee_zigzag_timer = random.randint(15, 45)
        agent.flee_angle_offset = random.choice([-50, -30, 30, 50])
    agent.flee_zigzag_timer -= 1

    desired_angle = (base_flee_angle + agent.flee_angle_offset) % 360
    diff = (desired_angle - agent.angle + 540) % 360 - 180

    # 轉向控制
    if abs(diff) > 10:
        if diff > 0:
            actions[4] = 1.0   # cw
        else:
            actions[5] = 1.0   # ccw

    # 前進與 Dash
    if abs(diff) < 90:
        actions[0] = 1.0
        if random.random() < 0.02 and agent.hp > 20 and agent.dash_cd == 0:
            actions[8] = 1.0   # Dash 逃生

    # 卡牆防呆：前方有牆就側移
    rad = math.radians(agent.angle)
    front_x = agent.x + math.cos(rad) * 40
    front_y = agent.y + math.sin(rad) * 40
    if env.is_wall(front_x, front_y):
        if agent.flee_angle_offset > 0:
            actions[3] = 1.0
        else:
            actions[2] = 1.0

    return actions


# ═══════════════════════════════════════════════════════
#  進階戰鬥 NPC（Stage 3 / 4）
# ═══════════════════════════════════════════════════════

def try_bullet_dodge(env, enemy, actions):
    """偵測到 AI / ally 子彈正朝向自己飛來時，有機率觸發 Dash 閃避。"""
    if enemy.dash_cd > 0 or enemy.hp <= GameConfig.DASH_COST_HP:
        return actions
    for p in env.projectiles:
        if p.owner.team_id == enemy.team_id:
            continue   # 不閃友軍子彈
        dx = enemy.x - p.x
        dy = enemy.y - p.y
        dist = math.hypot(dx, dy)
        if dist > 180:
            continue
        rad = math.radians(p.angle)
        bvx, bvy = math.cos(rad), math.sin(rad)
        dot = bvx * (dx / (dist + 1e-6)) + bvy * (dy / (dist + 1e-6))
        if dot > 0.7 and random.random() < 0.4:
            actions[8] = 1.0   # Dash
            break
    return actions


def get_flank_goal(env, enemy, target, base_orbit=220.0):
    """分配包抄角度，回傳目標位置 (goal_x, goal_y)。"""
    alive = env._alive_enemies()
    n = len(alive)
    if n <= 1:
        rad = math.radians(0.0)
        return (target.x + math.cos(rad) * base_orbit,
                target.y + math.sin(rad) * base_orbit)

    try:
        idx = alive.index(enemy)
    except ValueError:
        idx = 0

    if n == 2:
        angle_offsets = [-75.0, 75.0]
    else:
        angle_offsets = [-80.0, 0.0, 80.0]

    angle_offset = angle_offsets[min(idx, len(angle_offsets) - 1)]
    orbit = base_orbit
    target_rad = math.radians(angle_offset)
    goal_x = target.x + math.cos(target_rad) * orbit
    goal_y = target.y + math.sin(target_rad) * orbit

    for other in alive:
        if other is enemy:
            continue
        if math.hypot(other.x - goal_x, other.y - goal_y) < 90:
            orbit = 360.0
            goal_x = target.x + math.cos(target_rad) * orbit
            goal_y = target.y + math.sin(target_rad) * orbit
            break

    return goal_x, goal_y


def strafe_shoot_actions(env, enemy, target):
    """Z 字走位 + 開火 + 包抄。"""
    if not hasattr(enemy, 'strafe_timer'):
        enemy.strafe_timer = 0
        enemy.strafe_dir = random.choice([-1, 1])

    if enemy.strafe_timer <= 0:
        enemy.strafe_timer = random.randint(FPS, int(FPS * 2.5))
        enemy.strafe_dir = random.choice([-1, 1])
    enemy.strafe_timer -= 1

    actions = [0.0] * NUM_ACTIONS

    if enemy.strafe_dir > 0:
        actions[3] = 1.0
    else:
        actions[2] = 1.0

    alive = env._alive_enemies()
    dist = math.hypot(enemy.x - target.x, enemy.y - target.y)

    if len(alive) > 1:
        goal_x, goal_y = get_flank_goal(env, enemy, target)
        dx = goal_x - enemy.x
        dy = goal_y - enemy.y
        dist_goal = math.hypot(dx, dy)
        if dist_goal > 55:
            rad = math.radians(enemy.angle)
            fwd_x, fwd_y = math.cos(rad), math.sin(rad)
            rgt_x, rgt_y = math.cos(rad + math.pi / 2), math.sin(rad + math.pi / 2)
            mx, my = dx / dist_goal, dy / dist_goal
            fwd_dot = fwd_x * mx + fwd_y * my
            rgt_dot = rgt_x * mx + rgt_y * my
            if fwd_dot > 0.35:
                actions[0] = 1.0
            elif fwd_dot < -0.35:
                actions[1] = 1.0
            if abs(rgt_dot) > 0.45:
                if rgt_dot > 0:
                    actions[3] = 1.0
                    actions[2] = 0.0
                    enemy.strafe_dir = 1
                else:
                    actions[2] = 1.0
                    actions[3] = 0.0
                    enemy.strafe_dir = -1
    else:
        if dist > 260:
            actions[0] = 1.0
        elif dist < 130:
            actions[1] = 1.0

    aim = aim_and_shoot_actions(
        env, enemy, target,
        env.stage_spec.enemy_fire_rate,
        env.stage_spec.enemy_spread_deg,
        use_fov=False,
    )
    actions[4] = aim[4]
    actions[5] = aim[5]
    actions[6] = aim[6]

    return actions


def retreat_npc_actions(env, enemy, target):
    """撤退行為：面朝 AI 繼續開火 + 後退 + Z 字閃彈走位。"""
    if not hasattr(enemy, 'retreat_strafe_timer'):
        enemy.retreat_strafe_timer = 0
        enemy.retreat_strafe_dir = random.choice([-1, 1])
    if not hasattr(enemy, 'retreat_timer'):
        enemy.retreat_timer = FPS * NpcConfig.RETREAT_TIMER_SEC

    if enemy.retreat_strafe_timer <= 0:
        enemy.retreat_strafe_timer = random.randint(18, 45)
        enemy.retreat_strafe_dir = random.choice([-1, 1])
    enemy.retreat_strafe_timer -= 1
    enemy.retreat_timer = max(0, enemy.retreat_timer - 1)

    actions = [0.0] * NUM_ACTIONS
    actions[1] = 1.0
    if enemy.retreat_strafe_dir > 0:
        actions[3] = 1.0
    else:
        actions[2] = 1.0

    aim = aim_and_shoot_actions(
        env, enemy, target,
        env.stage_spec.enemy_fire_rate,
        env.stage_spec.enemy_spread_deg,
        use_fov=False,
    )
    actions[4] = aim[4]
    actions[5] = aim[5]
    actions[6] = aim[6]

    return actions


def can_agent_see_target(env, observer, target):
    """回傳 observer 是否看得到 target（在 FOV 且有 LOS）。"""
    rad = math.radians(observer.angle)
    fx, fy = math.cos(rad), math.sin(rad)
    rx, ry = math.cos(rad + math.pi / 2), math.sin(rad + math.pi / 2)
    dx = target.x - observer.x
    dy = target.y - observer.y
    ft = (dx * fx + dy * fy) / TILE_SIZE
    rt = (dx * rx + dy * ry) / TILE_SIZE
    dt = math.hypot(ft, rt)
    ang = math.degrees(math.atan2(rt, ft)) if dt > 0 else 0.0
    return (dt <= VIEW_RANGE and abs(ang) <= HALF_FOV
            and env.has_line_of_sight(observer.x, observer.y, target.x, target.y))


def _find_closest_target(env, enemy):
    """找出最近的存活敵對 learning agent"""
    candidates = [a for a in env.learning_agents
                  if a.alive() and a.team_id != enemy.team_id]
    if not candidates:
        return None
    return min(candidates, key=lambda a: math.hypot(a.x - enemy.x, a.y - enemy.y))


def combat_npc_actions(env, enemy):
    """Stage 3 / 4 的主要 NPC 行為狀態機。"""
    target = _find_closest_target(env, enemy)
    if target is None or not enemy.alive():
        return [0.0] * NUM_ACTIONS

    hp_ratio = enemy.hp / max(1, enemy.max_hp)
    dist = math.hypot(enemy.x - target.x, enemy.y - target.y)
    ai_sees_me = can_agent_see_target(env, target, enemy)

    if not hasattr(enemy, 'npc_state'):
        enemy.npc_state = 'wander'
    if not hasattr(enemy, 'retreat_timer'):
        enemy.retreat_timer = 0

    # 狀態轉換
    if enemy.npc_state == 'wander' and ai_sees_me:
        enemy.npc_state = 'combat'

    if enemy.npc_state == 'combat' and not ai_sees_me and hp_ratio >= 0.25:
        enemy.npc_state = 'wander'

    if enemy.npc_state == 'combat' and hp_ratio < 0.25:
        enemy.npc_state = 'retreat'
        enemy.retreat_timer = FPS * NpcConfig.RETREAT_TIMER_SEC

    if enemy.npc_state == 'retreat' and enemy.retreat_timer <= 0:
        enemy.npc_state = 'regen'

    if enemy.npc_state == 'regen' and dist < 280:
        enemy.npc_state = 'retreat'
        enemy.retreat_timer = FPS * NpcConfig.RETREAT_TIMER_SEC

    if enemy.npc_state in ('retreat', 'regen') and hp_ratio >= 0.25:
        enemy.npc_state = 'combat' if ai_sees_me else 'wander'

    # 行為執行
    if enemy.npc_state == 'wander':
        avoid = avoidance_actions(env, enemy, avoid_radius=NpcConfig.AVOID_RADIUS)
        actions = avoid if avoid else random_wander_actions(env, enemy, allow_shoot=False)
    elif enemy.npc_state == 'retreat':
        actions = retreat_npc_actions(env, enemy, target)
    elif enemy.npc_state == 'regen':
        actions = [0.0] * NUM_ACTIONS
    else:
        actions = strafe_shoot_actions(env, enemy, target)

    actions = try_bullet_dodge(env, enemy, actions)
    return actions


# ═══════════════════════════════════════════════════════
#  頂層 NPC 決策入口
# ═══════════════════════════════════════════════════════

def enemy_actions(env, enemy):
    if not enemy.alive():
        return [0.0] * NUM_ACTIONS

    candidates = [a for a in env.learning_agents
                  if a.alive() and a.team_id != enemy.team_id]
    if not candidates:
        return [0.0] * NUM_ACTIONS
    target = min(candidates,
                 key=lambda a: math.hypot(a.x - enemy.x, a.y - enemy.y))

    if enemy.bot_type == "dummy":
        return [0.0] * NUM_ACTIONS
    if enemy.bot_type == "self_play":
        return [0.0] * NUM_ACTIONS

    if env.stage_id == 2:
        return flee_actions(env, enemy, target)

    if env.stage_id in (3, 4):
        return combat_npc_actions(env, enemy)

    can_shoot = env.stage_spec.enemy_can_shoot or enemy.bot_type in ("turret_walk", "assault")

    if not can_shoot:
        avoid = avoidance_actions(env, enemy, avoid_radius=NpcConfig.AVOID_RADIUS)
        if avoid:
            return avoid
        return random_wander_actions(env, enemy, allow_shoot=False)
    else:
        actions = random_wander_actions(env, enemy, allow_shoot=False)
        aim = aim_and_shoot_actions(env, enemy, target, env.stage_spec.enemy_fire_rate, env.stage_spec.enemy_spread_deg, use_fov=False)

        if aim[4] > 0.0 or aim[5] > 0.0 or aim[6] > 0.0:
            actions[4] = aim[4]
            actions[5] = aim[5]
            actions[6] = aim[6]

        dist = math.hypot(target.x - enemy.x, target.y - enemy.y)
        if dist > 200:
            actions[0] = 1.0
            actions[1] = 0.0
        elif dist < 120:
            actions[0] = 0.0
            actions[1] = 1.0
        else:
            actions[0] = 0.0
            actions[1] = 0.0

        return actions


def teammate_actions(env, mate):
    if not mate.alive():
        return [0.0] * NUM_ACTIONS
    avoid = avoidance_actions(env, mate, avoid_radius=NpcConfig.AVOID_RADIUS)
    if avoid:
        return avoid
    return random_wander_actions(env, mate, allow_shoot=False)

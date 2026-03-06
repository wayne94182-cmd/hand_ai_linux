import math
import random

import numpy as np
import pygame

from game.config import (
    TILE_SIZE, COLS, ROWS, WIDTH, HEIGHT, FPS,
    VIEW_SIZE, VIEW_CENTER, VIEW_RANGE,
    FOV_DEGREES, HALF_FOV,
    NUM_ACTIONS, MAX_FRAMES,
    GameConfig, get_stage_spec,
)
from game.maps import MAPS, SMALL_MAPS
from game.fov import (
    _FOV_RC_NP, _FOV_FWD, _FOV_RIGHT, _RAY_FLAT, _RAY_OFFSETS, _RAY_LENGTHS,
    njit_has_line_of_sight, njit_compute_fov,
)
from game.entities import Agent, Projectile, Grenade


class GameEnv:
    def __init__(self, render_mode=False, stage_id=0, show_fov=True):
        self.render_mode = render_mode
        self.stage_id = stage_id
        self.show_fov = show_fov

        self.bullet_damage = GameConfig.BULLET_DAMAGE
        self.enemy_damage = GameConfig.BULLET_DAMAGE
        self.tie_penalty = GameConfig.TIE_PENALTY
        self.map_pool = MAPS

        self.screen = None
        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("AI 訓練環境")
            self.clock = pygame.time.Clock()
            pygame.font.init()
            self.font = pygame.font.SysFont(None, 24)

        self.set_stage(stage_id)

    def set_stage(self, stage_id):
        spec = get_stage_spec(stage_id)
        self.stage_id = stage_id
        self.stage_spec = spec
        self.enemy_damage = spec.enemy_damage
        self.bullet_damage = spec.bullet_damage
        self.tie_penalty = GameConfig.TIE_PENALTY

    def reset(self):
        if self.stage_id == 0 and random.random() < 0.8:
            self.grid = random.choice(SMALL_MAPS)
        else:
            self.grid = random.choice(self.map_pool)
        self.grid_np = np.array(self.grid, dtype=np.int8)
        empty_spots = [
            (c * TILE_SIZE + TILE_SIZE // 2, r * TILE_SIZE + TILE_SIZE // 2)
            for r in range(ROWS)
            for c in range(COLS)
            if self.grid[r, c] == 0
        ]

        n_agents = 1 + self.stage_spec.teammate_count + self.stage_spec.enemy_count
        spawns = []
        random.shuffle(empty_spots)
        for _ in range(n_agents):
            best_spot = empty_spots[0]
            if spawns:
                candidates = empty_spots[:20]  # Take up to 20 random available spots
                max_min_dist = -1
                for cand in candidates:
                    min_dist = min(math.hypot(cand[0]-s[0], cand[1]-s[1]) for s in spawns)
                    if min_dist > max_min_dist:
                        max_min_dist = min_dist
                        best_spot = cand
            spawns.append(best_spot)
            empty_spots.remove(best_spot)

        idx = 0
        self.ai_agent = Agent(spawns[idx][0], spawns[idx][1], (0, 140, 255), "ai", bot_type="learning")
        idx += 1

        self.team_agents = []
        for _ in range(self.stage_spec.teammate_count):
            a = Agent(spawns[idx][0], spawns[idx][1], (0, 220, 140), "ally", bot_type="wander")
            idx += 1
            self.team_agents.append(a)

        self.enemy_agents = []
        for i in range(self.stage_spec.enemy_count):
            bot_type = "runner" if self.stage_spec.enemy_mobile else "dummy"
            if self.stage_id == 3:
                bot_type = "turret_walk"
            elif self.stage_id == 4:
                bot_type = "assault"
            elif self.stage_id == 5:
                bot_type = "self_play"
            elif self.stage_id == 6 and i == 1:
                bot_type = "runner"
            elif self.stage_id == 6:
                bot_type = "assault"

            e = Agent(
                spawns[idx][0],
                spawns[idx][1],
                (255, 80, 80),
                f"enemy_{i}",
                bot_type=bot_type,
                infinite_ammo=(bot_type != "self_play"),
            )
            e.max_hp = self.stage_spec.enemy_hp
            e.hp = self.stage_spec.enemy_hp
            idx += 1
            self.enemy_agents.append(e)

        self.all_agents = [self.ai_agent] + self.team_agents + self.enemy_agents
        self.projectiles = []
        self.frame_count = 0
        self._last_info = {}
        return self.get_state()

    def get_state(self):
        return self._get_local_view(self.ai_agent)

    def is_wall(self, x, y):
        c = int(x // TILE_SIZE)
        r = int(y // TILE_SIZE)
        if 0 <= c < COLS and 0 <= r < ROWS:
            return self.grid[r, c] == 1
        return True

    def has_line_of_sight(self, x1, y1, x2, y2):
        return njit_has_line_of_sight(float(x1), float(y1), float(x2), float(y2), self.grid_np, float(TILE_SIZE), COLS, ROWS)

    def alive_agents(self):
        return [a for a in self.all_agents if a.alive()]

    def try_move_agent(self, agent, dx, dy):
        nx = agent.x + dx
        ny = agent.y + dy
        new_x = agent.x
        new_y = agent.y

        if not self.is_wall(nx, agent.y) and not self._collides_with_agent(nx, agent.y, agent):
            new_x = nx
        if not self.is_wall(agent.x, ny) and not self._collides_with_agent(agent.x, ny, agent):
            new_y = ny

        agent.x = new_x
        agent.y = new_y
        self._resolve_overlap(agent)

    def _collides_with_agent(self, x, y, me):
        for other in self.all_agents:
            if other is me or not other.alive():
                continue
            if math.hypot(x - other.x, y - other.y) < me.radius + other.radius:
                return True
        return False

    def _resolve_overlap(self, me):
        for other in self.all_agents:
            if other is me or not other.alive():
                continue
            dist = math.hypot(me.x - other.x, me.y - other.y)
            min_dist = me.radius + other.radius
            if 0 < dist < min_dist:
                push = (min_dist - dist + 1.0) / 2.0
                push_x = (me.x - other.x) / dist * push
                push_y = (me.y - other.y) / dist * push
                if not self.is_wall(me.x + push_x, me.y):
                    me.x += push_x
                if not self.is_wall(me.x, me.y + push_y):
                    me.y += push_y

    def _inject_value(self, channel, r_f, c_f, value):
        r0 = int(math.floor(r_f))
        c0 = int(math.floor(c_f))
        dr = r_f - r0
        dc = c_f - c0
        if 0 <= r0 < VIEW_SIZE and 0 <= c0 < VIEW_SIZE:
            channel[r0, c0] += value * (1.0 - dr) * (1.0 - dc)
        if 0 <= r0 < VIEW_SIZE and 0 <= c0 + 1 < VIEW_SIZE:
            channel[r0, c0 + 1] += value * (1.0 - dr) * dc
        if 0 <= r0 + 1 < VIEW_SIZE and 0 <= c0 < VIEW_SIZE:
            channel[r0 + 1, c0] += value * dr * (1.0 - dc)
        if 0 <= r0 + 1 < VIEW_SIZE and 0 <= c0 + 1 < VIEW_SIZE:
            channel[r0 + 1, c0 + 1] += value * dr * dc

    def _get_local_view(self, agent):
        rad = math.radians(agent.angle)
        fwd_x, fwd_y = math.cos(rad), math.sin(rad)
        rgt_x, rgt_y = math.cos(rad + math.pi / 2), math.sin(rad + math.pi / 2)
        ax, ay = agent.x, agent.y

        # Ch0: 地形, Ch1: 敵人雷達, Ch2: 子彈威脅, Ch3: 隊友全域層
        view = np.zeros((4, VIEW_SIZE, VIEW_SIZE), dtype=np.float32)

        view[0] = njit_compute_fov(
            float(ax),
            float(ay),
            float(fwd_x),
            float(fwd_y),
            float(rgt_x),
            float(rgt_y),
            self.grid_np,
            _FOV_RC_NP,
            _FOV_FWD,
            _FOV_RIGHT,
            _RAY_FLAT,
            _RAY_OFFSETS,
            _RAY_LENGTHS,
            float(TILE_SIZE),
            COLS,
            ROWS,
            VIEW_SIZE,
        )

        for enemy in self.enemy_agents:
            if not enemy.alive():
                continue
            dx = enemy.x - ax
            dy = enemy.y - ay
            ft = (dx * fwd_x + dy * fwd_y) / TILE_SIZE
            rt = (dx * rgt_x + dy * rgt_y) / TILE_SIZE
            dt = math.hypot(ft, rt)
            ang = math.degrees(math.atan2(rt, ft)) if dt > 0 else 0.0
            if dt <= VIEW_RANGE and abs(ang) <= HALF_FOV and self.has_line_of_sight(ax, ay, enemy.x, enemy.y):
                dr = (ft + rt) / 1.41421356
                dc = (ft - rt) / 1.41421356
                self._inject_value(view[1], VIEW_CENTER + dr, VIEW_CENTER + dc, 1.0)

        for p in self.projectiles:
            dx = p.x - ax
            dy = p.y - ay
            ft = (dx * fwd_x + dy * fwd_y) / TILE_SIZE
            rt = (dx * rgt_x + dy * rgt_y) / TILE_SIZE
            dt = math.hypot(ft, rt)
            ang = math.degrees(math.atan2(rt, ft)) if dt > 0 else 0.0
            if dt <= VIEW_RANGE and abs(ang) <= HALF_FOV and self.has_line_of_sight(ax, ay, p.x, p.y):
                val = 1.0 if p.owner.team.startswith("enemy") else -1.0
                dr = (ft + rt) / 1.41421356
                dc = (ft - rt) / 1.41421356
                self._inject_value(view[2], VIEW_CENTER + dr, VIEW_CENTER + dc, val)

        alive_mates = [m for m in self.team_agents if m.alive()]
        for m in alive_mates:
            dx = m.x - ax
            dy = m.y - ay
            mft = (dx * fwd_x + dy * fwd_y) / TILE_SIZE
            mrt = (dx * rgt_x + dy * rgt_y) / TILE_SIZE
            dr = (mft + mrt) / 1.41421356
            dc = (mft - mrt) / 1.41421356
            r_f = np.clip(VIEW_CENTER + dr, 0.0, VIEW_SIZE - 1.001)
            c_f = np.clip(VIEW_CENTER + dc, 0.0, VIEW_SIZE - 1.001)
            # Ch3: 隊友全域層，不受視角與視距限制
            self._inject_value(view[3], r_f, c_f, 1.0)

        norm_ft = 0.0
        norm_rt = 0.0
        if alive_mates:
            closest_mate = min(alive_mates, key=lambda m: math.hypot(m.x - ax, m.y - ay))
            dx = closest_mate.x - ax
            dy = closest_mate.y - ay
            cft = (dx * fwd_x + dy * fwd_y) / TILE_SIZE
            crt = (dx * rgt_x + dy * rgt_y) / TILE_SIZE

            norm_ft = float(np.clip(cft / COLS, -1.0, 1.0))
            norm_rt = float(np.clip(crt / ROWS, -1.0, 1.0))

        ammo_ratio = agent.ammo / max(1, agent.max_ammo)
        reload_ratio = agent.reload_timer / max(1, agent.reload_delay) if agent.ammo == 0 else 0.0
        hp_ratio = agent.hp / max(1, agent.max_hp)
        dash_ratio = agent.dash_cd / 160.0
        regen_ratio = agent.regen_timer / 240.0
        hit_marker = 1.0 if getattr(agent, "hit_marker_timer", 0) > 0 else 0.0
        has_teammate = 1.0 if alive_mates else 0.0

        scalars = np.array([
            ammo_ratio, reload_ratio, hp_ratio, dash_ratio,
            regen_ratio, hit_marker, norm_ft, norm_rt, has_teammate
        ], dtype=np.float32)

        return view, scalars

    def _random_wander_actions(self, agent, allow_shoot=False):
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

    def _aim_and_shoot_actions(self, shooter, target, fire_rate, spread_deg, use_fov=True):
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
        if fire_rate > 0 and dist_tiles <= VIEW_RANGE and angle_ok and self.has_line_of_sight(shooter.x, shooter.y, target.x, target.y):
            jitter = random.uniform(0.0, spread_deg)
            signed = jitter if random.random() < 0.5 else -jitter
            shooter.angle = (desired + signed) % 360
            actions[6] = 1.0
        return actions

    def _avoidance_actions(self, agent, avoid_radius=150.0):
        rx, ry = 0.0, 0.0
        count = 0
        for other in self.all_agents:
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
        actions[0] = 1.0 # Forward
        
        if abs(diff) > 15:
            if diff > 0:
                actions[4] = 1.0
            else:
                actions[5] = 1.0
        return actions

    def _flee_actions(self, agent, target):
        actions = [0.0] * NUM_ACTIONS

        has_los = self.has_line_of_sight(agent.x, agent.y, target.x, target.y)
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
                # 記憶歸零：已安全，回到普通遊蕩
                return self._random_wander_actions(agent, allow_shoot=False)

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
                actions[4] = 1.0  # 順時針轉 (cw)
            else:
                actions[5] = 1.0  # 逆時針轉 (ccw)

        # 前進與 Dash
        if abs(diff) < 90:
            actions[0] = 1.0
            if random.random() < 0.02 and agent.hp > 20 and agent.dash_cd == 0:
                actions[8] = 1.0  # Dash 逃生

        # 卡牆防呆：前方有牆就側移
        rad = math.radians(agent.angle)
        front_x = agent.x + math.cos(rad) * 40
        front_y = agent.y + math.sin(rad) * 40
        if self.is_wall(front_x, front_y):
            if agent.flee_angle_offset > 0:
                actions[3] = 1.0  # 往右平移
            else:
                actions[2] = 1.0  # 往左平移

        return actions

    # ═══════════════════════════════════════════════════════
    #  Stage 3 / 4  進階戰鬥 NPC 系統
    # ═══════════════════════════════════════════════════════

    def _try_bullet_dodge(self, enemy, actions):
        """
        偵測到 AI / ally 子彈正朝向自己飛來時，
        有機率觸發 Dash 閃避（任意 NPC 狀態皆可觸發）。
        """
        if enemy.dash_cd > 0 or enemy.hp <= GameConfig.DASH_COST_HP:
            return actions
        for p in self.projectiles:
            if p.owner.team.startswith("enemy"):   # 只閃 AI/ally 的子彈
                continue
            dx = enemy.x - p.x
            dy = enemy.y - p.y
            dist = math.hypot(dx, dy)
            if dist > 180:
                continue
            rad = math.radians(p.angle)
            bvx, bvy = math.cos(rad), math.sin(rad)
            # 子彈飛行方向 dot 「子彈→敵人」方向，> 0.7 = 對著臉射
            dot = bvx * (dx / (dist + 1e-6)) + bvy * (dy / (dist + 1e-6))
            if dot > 0.7 and random.random() < 0.4:
                actions[8] = 1.0   # Dash
                break
        return actions

    def _get_flank_goal(self, enemy, target, base_orbit=220.0):
        """
        根據存活敵人數量與敵人在 alive list 中的索引，
        分配包抄角度（世界座標），回傳目標位置 (goal_x, goal_y)。
        若目標位置已被同伴佔用，自動切換到外軌（350px）。
        """
        alive = self._alive_enemies()
        n = len(alive)
        if n <= 1:
            rad = math.radians(0.0)
            return (target.x + math.cos(rad) * base_orbit,
                    target.y + math.sin(rad) * base_orbit)

        try:
            idx = alive.index(enemy)
        except ValueError:
            idx = 0

        # 依人數分配角度偏移（相對於世界 x 軸正方向）
        if n == 2:
            angle_offsets = [-75.0, 75.0]
        else:
            angle_offsets = [-80.0, 0.0, 80.0]

        angle_offset = angle_offsets[min(idx, len(angle_offsets) - 1)]
        orbit = base_orbit
        target_rad = math.radians(angle_offset)
        goal_x = target.x + math.cos(target_rad) * orbit
        goal_y = target.y + math.sin(target_rad) * orbit

        # 若目標位置已被其他同伴佔據 → 切換外軌
        for other in alive:
            if other is enemy:
                continue
            if math.hypot(other.x - goal_x, other.y - goal_y) < 90:
                orbit = 360.0
                goal_x = target.x + math.cos(target_rad) * orbit
                goal_y = target.y + math.sin(target_rad) * orbit
                break

        return goal_x, goal_y

    def _strafe_shoot_actions(self, enemy, target):
        """
        Z 字走位：面朝 AI 持續開火 + 左右 Strafe（定時隨機換向）。
        若有存活同伴，加入包抄位置修正（Flanking）；
        若無同伴，按距離前進/後退。
        """
        # ── 初始化 strafe 屬性 ──
        if not hasattr(enemy, 'strafe_timer'):
            enemy.strafe_timer = 0
            enemy.strafe_dir = random.choice([-1, 1])

        if enemy.strafe_timer <= 0:
            enemy.strafe_timer = random.randint(FPS, int(FPS * 2.5))
            enemy.strafe_dir = random.choice([-1, 1])
        enemy.strafe_timer -= 1

        actions = [0.0] * NUM_ACTIONS

        # ── Strafe 左右 ──
        if enemy.strafe_dir > 0:
            actions[3] = 1.0   # rt
        else:
            actions[2] = 1.0   # lt

        alive = self._alive_enemies()
        dist = math.hypot(enemy.x - target.x, enemy.y - target.y)

        if len(alive) > 1:
            # ── 包抄位置修正 ──
            goal_x, goal_y = self._get_flank_goal(enemy, target)
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
                # 前後修正（接近/遠離包抄點）
                if fwd_dot > 0.35:
                    actions[0] = 1.0
                elif fwd_dot < -0.35:
                    actions[1] = 1.0
                # 左右修正：包抄拉力大於 Strafe 閾值時覆蓋 Strafe
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
            # ── 單獨時維持距離 ──
            if dist > 260:
                actions[0] = 1.0
            elif dist < 130:
                actions[1] = 1.0

        # ── 瞄準並開火（覆蓋轉向） ──
        aim = self._aim_and_shoot_actions(
            enemy, target,
            self.stage_spec.enemy_fire_rate,
            self.stage_spec.enemy_spread_deg,
            use_fov=False,
        )
        actions[4] = aim[4]
        actions[5] = aim[5]
        actions[6] = aim[6]

        return actions

    def _retreat_npc_actions(self, enemy, target):
        """
        撤退行為：面朝 AI 繼續開火 + 後退 + Z 字閃彈走位。
        retreat_timer 每幀遞減，歸 0 後由狀態機切換至 regen。
        """
        if not hasattr(enemy, 'retreat_strafe_timer'):
            enemy.retreat_strafe_timer = 0
            enemy.retreat_strafe_dir = random.choice([-1, 1])
        if not hasattr(enemy, 'retreat_timer'):
            enemy.retreat_timer = FPS * 4

        # 蛇行換向
        if enemy.retreat_strafe_timer <= 0:
            enemy.retreat_strafe_timer = random.randint(18, 45)
            enemy.retreat_strafe_dir = random.choice([-1, 1])
        enemy.retreat_strafe_timer -= 1
        enemy.retreat_timer = max(0, enemy.retreat_timer - 1)

        actions = [0.0] * NUM_ACTIONS
        actions[1] = 1.0   # 後退
        if enemy.retreat_strafe_dir > 0:
            actions[3] = 1.0
        else:
            actions[2] = 1.0

        # 面朝 AI 並繼續開火
        aim = self._aim_and_shoot_actions(
            enemy, target,
            self.stage_spec.enemy_fire_rate,
            self.stage_spec.enemy_spread_deg,
            use_fov=False,
        )
        actions[4] = aim[4]
        actions[5] = aim[5]
        actions[6] = aim[6]

        return actions

    def _ai_can_see_enemy(self, enemy):
        """回傳 AI 是否看得到這個敵人（在 FOV 且有 LOS）。"""
        ai = self.ai_agent
        rad = math.radians(ai.angle)
        fx, fy = math.cos(rad), math.sin(rad)
        rx, ry = math.cos(rad + math.pi / 2), math.sin(rad + math.pi / 2)
        dx = enemy.x - ai.x
        dy = enemy.y - ai.y
        ft = (dx * fx + dy * fy) / TILE_SIZE
        rt = (dx * rx + dy * ry) / TILE_SIZE
        dt = math.hypot(ft, rt)
        ang = math.degrees(math.atan2(rt, ft)) if dt > 0 else 0.0
        return (dt <= VIEW_RANGE and abs(ang) <= HALF_FOV
                and self.has_line_of_sight(ai.x, ai.y, enemy.x, enemy.y))

    def _combat_npc_actions(self, enemy):
        """
        Stage 3 / 4 的主要 NPC 行為狀態機。
        ┌─────────┬──────────────────────────────────────────────────────┐
        │ 狀態    │ 觸發條件                                             │
        ├─────────┼──────────────────────────────────────────────────────┤
        │ wander  │ 預設；AI 看不到這個敵人                             │
        │ combat  │ AI 看到這個敵人後進入；hp≥25%                       │
        │ retreat │ hp<25%；或 regen 時 AI 再次逼近（dist<280）         │
        │ regen   │ retreat_timer 倒數至 0（已拉開足夠距離）            │
        └─────────┴──────────────────────────────────────────────────────┘
        retreat/regen → combat：hp 恢復到 ≥25% 且 AI 仍看得到
        retreat/regen → wander：hp 恢復到 ≥25% 且 AI 看不到
        """
        target = self.ai_agent
        if not target.alive() or not enemy.alive():
            return [0.0] * NUM_ACTIONS

        hp_ratio = enemy.hp / max(1, enemy.max_hp)
        dist = math.hypot(enemy.x - target.x, enemy.y - target.y)
        ai_sees_me = self._ai_can_see_enemy(enemy)

        # ── 初始化 ──
        if not hasattr(enemy, 'npc_state'):
            enemy.npc_state = 'wander'
        if not hasattr(enemy, 'retreat_timer'):
            enemy.retreat_timer = 0

        # ── 狀態轉換 ──
        # wander → combat：AI 看到我了
        if enemy.npc_state == 'wander' and ai_sees_me:
            enemy.npc_state = 'combat'

        # combat → wander：AI 看不到我（且血量正常）
        if enemy.npc_state == 'combat' and not ai_sees_me and hp_ratio >= 0.25:
            enemy.npc_state = 'wander'

        # combat → retreat：血量不足
        if enemy.npc_state == 'combat' and hp_ratio < 0.25:
            enemy.npc_state = 'retreat'
            enemy.retreat_timer = FPS * 4

        # retreat → regen：撤退倒數結束
        if enemy.npc_state == 'retreat' and enemy.retreat_timer <= 0:
            enemy.npc_state = 'regen'

        # regen → retreat：AI 追來
        if enemy.npc_state == 'regen' and dist < 280:
            enemy.npc_state = 'retreat'
            enemy.retreat_timer = FPS * 4

        # retreat / regen → combat or wander：血量恢復
        if enemy.npc_state in ('retreat', 'regen') and hp_ratio >= 0.25:
            enemy.npc_state = 'combat' if ai_sees_me else 'wander'

        # ── 行為執行 ──
        if enemy.npc_state == 'wander':
            # 原有的隨機走位
            avoid = self._avoidance_actions(enemy, avoid_radius=150.0)
            actions = avoid if avoid else self._random_wander_actions(enemy, allow_shoot=False)
        elif enemy.npc_state == 'retreat':
            actions = self._retreat_npc_actions(enemy, target)
        elif enemy.npc_state == 'regen':
            actions = [0.0] * NUM_ACTIONS      # 靜止發呆，自動回血
        else:
            actions = self._strafe_shoot_actions(enemy, target)

        # ── 子彈閃避（任意狀態都可觸發） ──
        actions = self._try_bullet_dodge(enemy, actions)

        return actions

    # ═══════════════════════════════════════════════════════

    def _enemy_actions(self, enemy):
        if not enemy.alive():
            return [0.0] * NUM_ACTIONS

        target = self.ai_agent

        if enemy.bot_type == "dummy":
            return [0.0] * NUM_ACTIONS
        if enemy.bot_type == "self_play":
            return [0.0] * NUM_ACTIONS

        # Stage 2：追獵期，敵人會逃跑（只有被 AI 看到才觸發）
        if self.stage_id == 2:
            return self._flee_actions(enemy, target)

        # Stage 3 / 4：進階戰鬥 NPC
        if self.stage_id in (3, 4):
            return self._combat_npc_actions(enemy)

        can_shoot = self.stage_spec.enemy_can_shoot or enemy.bot_type in ("turret_walk", "assault")

        if not can_shoot:
            # 不會開槍的: 偏好往沒有人的地方走
            avoid = self._avoidance_actions(enemy, avoid_radius=150.0)
            if avoid:
                return avoid
            return self._random_wander_actions(enemy, allow_shoot=False)
        else:
            # 會開槍的: 往 AI 走，但保持一定社交距離
            actions = self._random_wander_actions(enemy, allow_shoot=False)
            aim = self._aim_and_shoot_actions(enemy, target, self.stage_spec.enemy_fire_rate, self.stage_spec.enemy_spread_deg, use_fov=False)
            
            if aim[4] > 0.0 or aim[5] > 0.0 or aim[6] > 0.0:
                actions[4] = aim[4]
                actions[5] = aim[5]
                actions[6] = aim[6]
                
            dist = math.hypot(target.x - enemy.x, target.y - enemy.y)
            if dist > 200:
                actions[0] = 1.0  # 前進
                actions[1] = 0.0
            elif dist < 120:
                actions[0] = 0.0
                actions[1] = 1.0  # 後退保持距離
            else:
                actions[0] = 0.0  # 範圍內停步開火
                actions[1] = 0.0
                
            return actions

    def _teammate_actions(self, mate):
        if not mate.alive():
            return [0.0] * NUM_ACTIONS
        avoid = self._avoidance_actions(mate, avoid_radius=150.0)
        if avoid:
            return avoid
        return self._random_wander_actions(mate, allow_shoot=False)

    def _radar_visible_any_enemy(self, agent):
        rad = math.radians(agent.angle)
        fx, fy = math.cos(rad), math.sin(rad)
        rx, ry = math.cos(rad + math.pi / 2), math.sin(rad + math.pi / 2)
        for enemy in self.enemy_agents:
            if not enemy.alive():
                continue
            dx, dy = enemy.x - agent.x, enemy.y - agent.y
            ft = (dx * fx + dy * fy) / TILE_SIZE
            rt = (dx * rx + dy * ry) / TILE_SIZE
            dist_t = math.hypot(ft, rt)
            ang_t = math.degrees(math.atan2(rt, ft)) if dist_t > 0 else 0.0
            if dist_t <= VIEW_RANGE and abs(ang_t) <= HALF_FOV and self.has_line_of_sight(agent.x, agent.y, enemy.x, enemy.y):
                return True
        return False

    def _alive_enemies(self):
        return [e for e in self.enemy_agents if e.alive()]

    def _alive_allies(self):
        return [a for a in [self.ai_agent] + self.team_agents if a.alive()]

    def _single_step(self, ai_action, enemy_ai_action=None):
        self.frame_count += 1
        reward = 0.0

        # 從 stage_spec 讀取本階段的動作誤差比例設定
        _move_noise_pct = float(getattr(self.stage_spec, "move_noise_pct", 0.0))
        _rot_noise_pct  = float(getattr(self.stage_spec, "rotation_noise_pct", 0.0))

        shot, dash_reward = self.ai_agent.apply_actions(ai_action, self, move_noise_pct=_move_noise_pct, rotation_noise_pct=_rot_noise_pct)
        if shot:
            reward -= GameConfig.SHOOT_PENALTY
        reward += dash_reward

        for mate in self.team_agents:
            mate.apply_actions(self._teammate_actions(mate), self)  # NPC 不加噪聲

        for i, enemy in enumerate(self.enemy_agents):
            if self.stage_id == 5 and i == 0 and enemy_ai_action is not None:
                enemy.apply_actions(enemy_ai_action, self)  # 對戰 AI 不加噪聲
            else:
                enemy.apply_actions(self._enemy_actions(enemy), self)  # NPC 不加噪聲

        enemy_in_sight = False
        best_angle = 180.0
        ai = self.ai_agent
        rad = math.radians(ai.angle)
        fx, fy = math.cos(rad), math.sin(rad)
        rx, ry = math.cos(rad + math.pi / 2), math.sin(rad + math.pi / 2)
        for enemy in self.enemy_agents:
            if not enemy.alive():
                continue
            dx = enemy.x - ai.x
            dy = enemy.y - ai.y
            ft = (dx * fx + dy * fy) / TILE_SIZE
            rt = (dx * rx + dy * ry) / TILE_SIZE
            dt = math.hypot(ft, rt)
            ang = math.degrees(math.atan2(rt, ft)) if dt > 0 else 0.0
            if dt <= VIEW_RANGE and abs(ang) <= HALF_FOV and self.has_line_of_sight(ai.x, ai.y, enemy.x, enemy.y):
                enemy_in_sight = True
                best_angle = min(best_angle, abs(ang))
        if enemy_in_sight:
            reward += GameConfig.RADAR_REWARD
            
            if self.stage_id in (0, 1):
                # --- 新的鎖定瞄準邏輯 ---
                AIM_TOLERANCE = 15.0  # 目標必須在準星正負 15 度以內
                REQUIRED_FRAMES = 5   # 必須連續維持 5 個 Frame

                if best_angle <= AIM_TOLERANCE:
                    self.ai_agent.aim_frames += 1
                else:
                    self.ai_agent.aim_frames = 0  # 角度偏掉，計時器歸零

                # 當連續瞄準達到指定幀數時才給獎勵
                if self.ai_agent.aim_frames >= REQUIRED_FRAMES:
                    reward += GameConfig.aim_reward  # 因為條件變嚴格了，可以把單次給予的獎勵調高一點
                    
                    # 關鍵：給完獎勵後強制歸零，這樣 AI 為了拿下次獎勵會被迫重新尋找或開槍
                    # 避免它永遠停在原地不開槍只刷鎖定分數
                    self.ai_agent.aim_frames = 0
        else:
            # 敵人不在視野內，計時器也要歸零
            self.ai_agent.aim_frames = 0

        for enemy in self._alive_enemies():
            if math.hypot(self.ai_agent.x - enemy.x, self.ai_agent.y - enemy.y) < self.ai_agent.radius + enemy.radius + 1:
                reward -= GameConfig.COLLISION_PENALTY

        for p in self.projectiles[:]:
            hit_someone = False
            for ag in self.all_agents:
                if (not ag.alive()) or (ag is p.owner):
                    continue
                    
                # ★ 新增：防護網邏輯 (關閉友軍傷害)
                owner_is_good = p.owner.team in ("ai", "ally")
                ag_is_good = ag.team in ("ai", "ally")
                
                # 如果雙方都是我方，或者雙方都是敵方，子彈直接穿透不造成傷害
                if (owner_is_good and ag_is_good) or (p.owner.team.startswith("enemy") and ag.team.startswith("enemy")):
                    continue

                # 原本的碰撞扣血與給分邏輯
                if math.hypot(p.x - ag.x, p.y - ag.y) < ag.radius + p.radius:
                    ag.hp -= p.damage
                    if ag is self.ai_agent:
                        reward -= GameConfig.DAMAGE_PENALTY
                    if p.owner is self.ai_agent and ag.team.startswith("enemy"):
                        reward += GameConfig.HIT_REWARD
                        self.ai_agent.hit_marker_timer = 3
                        if not ag.alive():
                            reward += GameConfig.NPC_KILL_REWARD
                            
                    if p in self.projectiles:
                        self.projectiles.remove(p)
                        
                    hit_someone = True
                    break
                    
            if hit_someone:
                continue

            p.update()
            if self.is_wall(p.x, p.y) or p.life <= 0:
                if p in self.projectiles:
                    self.projectiles.remove(p)
                continue

        if self.stage_id in (0, 1, 2, 3, 4):
            reward -= GameConfig.NPC_SURVIVAL_COST * len(self._alive_enemies())
        else:
            reward -= GameConfig.SURVIVAL_COST

        done = False
        ai_win = False
        ai_lost = False

        ai_alive = self.ai_agent.alive()
        enemies_alive = len(self._alive_enemies()) > 0
        allies_alive = len(self._alive_allies()) > 0

        if self.stage_id in (0, 1, 2, 3, 4):
            time_out = self.frame_count >= self.stage_spec.max_frames
            if not ai_alive or not enemies_alive or time_out:
                done = True
                if time_out and enemies_alive:
                    reward -= GameConfig.ALIVE_NPC_PENALTY
                
                if ai_alive and not enemies_alive:
                    ai_win = True
                elif (not ai_alive) and enemies_alive:
                    ai_lost = True
                    reward -= GameConfig.LOSE_PENALTY
                else:
                    reward -= self.tie_penalty
                    
        elif self.stage_id == 5:
            if not ai_alive or not enemies_alive or self.frame_count >= self.stage_spec.max_frames:
                done = True
                if ai_alive and not enemies_alive:
                    ai_win = True
                    reward += GameConfig.WIN_REWARD
                elif (not ai_alive) and enemies_alive:
                    ai_lost = True
                    reward -= GameConfig.LOSE_PENALTY
                else:
                    reward -= self.tie_penalty

        elif self.stage_id == 6:
            enemies_team_dead = not enemies_alive
            allies_team_dead = not allies_alive
            if enemies_team_dead or allies_team_dead or self.frame_count >= self.stage_spec.max_frames:
                done = True
                if enemies_team_dead and not allies_team_dead:
                    ai_win = True
                    reward += GameConfig.WIN_REWARD
                elif allies_team_dead and not enemies_team_dead:
                    ai_lost = True
                    reward -= GameConfig.LOSE_PENALTY
                else:
                    reward -= self.tie_penalty

        info = {
            "stage_id": self.stage_id,
            "stage_name": self.stage_spec.name,
            "ai_win": ai_win,
            "ai_lost": ai_lost,
            "ai_alive": ai_alive,
            "enemy_alive_count": len(self._alive_enemies()),
            "ally_alive_count": len([a for a in self.team_agents if a.alive()]),
            "ai_kill_target": ai_win,
            "kill_count": self.stage_spec.enemy_count - len(self._alive_enemies()),
        }

        self._last_info = info
        return self.get_state(), reward, done, info

    def step(self, ai_action, enemy_ai_action=None, frame_skip=1):
        total = 0.0
        done = False
        state = None
        info = {}
        for _ in range(frame_skip):
            if done:
                break
            state, rew, done, info = self._single_step(ai_action, enemy_ai_action=enemy_ai_action)
            total += rew
        return state, total, done, info

    def _draw_agent(self, a):
        pygame.draw.circle(self.screen, a.color, (int(a.x), int(a.y)), a.radius)
        rad = math.radians(a.angle)
        ex = a.x + math.cos(rad) * a.radius * 1.5
        ey = a.y + math.sin(rad) * a.radius * 1.5
        pygame.draw.line(self.screen, (255, 255, 0), (a.x, a.y), (ex, ey), 2)

    def _draw_fov(self, a):
        if not self.show_fov:
            return
        rad = math.radians(a.angle)
        fwd_x, fwd_y = math.cos(rad), math.sin(rad)
        rgt_x, rgt_y = math.cos(rad + math.pi / 2), math.sin(rad + math.pi / 2)
        
        view_r = float(VIEW_RANGE)
        
        # 多邊形頂點列表，首尾都接 AI 本體座標
        pts = [(a.x, a.y)]
        
        steps = 60
        for i in range(steps + 1):
            # 相對於 Agent 面向的角度 (從 -65度 到 +65度)
            deg_rel = -HALF_FOV + (FOV_DEGREES * i / steps)
            rad_rel = math.radians(deg_rel)
            
            cos_rel = math.cos(rad_rel)
            sin_rel = math.sin(rad_rel)
            
            # 將局部座標轉換為旋轉後的世界座標
            ft_val = view_r * cos_rel
            rt_val = view_r * sin_rel
            wx = a.x + (fwd_x * ft_val + rgt_x * rt_val) * TILE_SIZE
            wy = a.y + (fwd_y * ft_val + rgt_y * rt_val) * TILE_SIZE
            pts.append((wx, wy))
            
        pts.append((a.x, a.y))
        
        if len(pts) > 2:
            # 畫出裁切後的扇形邊界
            pygame.draw.lines(self.screen, (80, 220, 255), True, pts, 1)

    def render(self, info=""):
        if not self.render_mode:
            return
        self.screen.fill((20, 20, 30))

        for r in range(ROWS):
            for c in range(COLS):
                if self.grid[r, c] == 1:
                    pygame.draw.rect(self.screen, (100, 100, 120), (c * TILE_SIZE, r * TILE_SIZE, TILE_SIZE, TILE_SIZE))

        for p in self.projectiles:
            p.draw(self.screen)

        for a in self.all_agents:
            if a.alive():
                self._draw_agent(a)

        self._draw_fov(self.ai_agent)

        line1 = f"Stage {self.stage_id} {self.stage_spec.name} | AI HP:{self.ai_agent.hp} Ammo:{self.ai_agent.ammo}"
        self.screen.blit(self.font.render(line1, True, (255, 255, 255)), (10, 8))

        line2 = f"EnemyAlive:{len(self._alive_enemies())} AllyAlive:{len([a for a in self.team_agents if a.alive()])}"
        self.screen.blit(self.font.render(line2, True, (255, 255, 255)), (10, 30))

        if info:
            self.screen.blit(self.font.render(info, True, (255, 220, 50)), (10, HEIGHT - 28))

        pygame.display.flip()
        self.clock.tick(FPS)

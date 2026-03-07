"""
game/env.py — GameEnv 全面重寫
支援動態 N 個 learning agents、新觀測通道（6ch + 22 scalars）、
武器/道具/手榴彈系統、聲音波紋、NPC 模組化。
"""
import math
import math as _math
import random
from typing import List, Tuple, Optional

import numpy as np
import pygame

from game.config import (
    TILE_SIZE, COLS, ROWS, WIDTH, HEIGHT, FPS,
    VIEW_SIZE, VIEW_CENTER, VIEW_RANGE,
    FOV_DEGREES, HALF_FOV,
    NUM_ACTIONS, MAX_FRAMES,
    GameConfig, get_stage_spec,
)
from game.maps import MAPS, SMALL_MAPS, MEDIUM_MAPS, LARGE_MAPS
from game.fov import (
    _FOV_RC_NP, _FOV_FWD, _FOV_RIGHT, _RAY_FLAT, _RAY_OFFSETS, _RAY_LENGTHS,
    njit_has_line_of_sight, njit_compute_fov,
    njit_compute_fov_standard, njit_compute_fov_sniper,
    get_fov_tables,
)
from game.entities import Agent, Projectile, Grenade
from game.items import (
    PISTOL, RIFLE, SHOTGUN, SNIPER, WEAPON_TYPES,
    GroundItem, try_auto_pickup,
)
from game.audio import (
    SoundWave,
    create_footstep_wave, create_reload_wave,
    create_gunshot_wave, create_explosion_wave,
    render_sound_channel,
)
from game.npc import enemy_actions, teammate_actions

# ─── 常數 ────────────────────────────────────────────
NUM_CHANNELS = 6
NUM_SCALARS = 22

# 散彈槍彈片角度偏移
_SHOTGUN_OFFSETS = [-30.0, -15.0, 0.0, 15.0, 30.0]

# 道具生成權重
_WEAPON_WEIGHTS = [0.3, 0.4, 0.2, 0.1]  # PISTOL, RIFLE, SHOTGUN, SNIPER
_ITEM_TYPE_WEIGHTS = [0.5, 0.3, 0.2]     # weapon, medkit, grenade

_MAP_POOL = {"small": SMALL_MAPS, "medium": MEDIUM_MAPS, "large": LARGE_MAPS}

def _sample_log_uniform(lo: float, hi: float) -> float:
    if lo == hi:
        return lo
    return _math.exp(random.uniform(_math.log(lo), _math.log(hi)))


class GameEnv:
    def __init__(self, render_mode=False, stage_id=0, show_fov=True,
                 n_learning_agents=1):
        self.render_mode = render_mode
        self.stage_id = stage_id
        self.show_fov = show_fov
        self.n_learning_agents = n_learning_agents

        self.bullet_damage = GameConfig.BULLET_DAMAGE
        self.enemy_damage = GameConfig.BULLET_DAMAGE
        self.tie_penalty = GameConfig.TIE_PENALTY
        self.map_pool = MAPS

        # 列表管理
        self.learning_agents: List[Agent] = []
        self.team_agents: List[Agent] = []
        self.enemy_agents: List[Agent] = []
        self.all_agents: List[Agent] = []
        self.projectiles: List[Projectile] = []
        self.grenades_list: List[Grenade] = []
        self.ground_items: List[GroundItem] = []
        self.sound_waves: List[SoundWave] = []

        self.frame_count = 0
        self._prev_frame = 0
        self._last_info = {}

        self.screen = None
        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("AI 訓練環境")
            self.clock = pygame.time.Clock()
            pygame.font.init()
            self.font = pygame.font.SysFont(None, 24)

        self.set_stage(stage_id)

    # ── 向後相容 property ──
    @property
    def ai_agent(self) -> Agent:
        return self.learning_agents[0]

    # ── Stage 設定 ──

    def set_stage(self, stage_id):
        spec = get_stage_spec(stage_id)
        self.stage_id = stage_id
        self.stage_spec = spec
        self.enemy_damage = spec.enemy_damage
        self.bullet_damage = spec.bullet_damage
        self.tie_penalty = GameConfig.TIE_PENALTY

    # ── Reset ──

    def reset(self):
        # 依 map_pool_key 選擇地圖池 (支援以 '+' 分隔多個地圖池)
        pool = []
        keys = self.stage_spec.map_pool_key.split('+')
        for k in keys:
            pool.extend(_MAP_POOL.get(k.strip(), SMALL_MAPS))
        if not pool:
            pool = SMALL_MAPS
        self.grid = random.choice(pool)
        grid_rows, grid_cols = self.grid.shape
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.grid_np = np.array(self.grid, dtype=np.int8)
        empty_spots = [
            (c * TILE_SIZE + TILE_SIZE // 2, r * TILE_SIZE + TILE_SIZE // 2)
            for r in range(grid_rows)
            for c in range(grid_cols)
            if self.grid[r, c] == 0
        ]

        n_la = self.n_learning_agents
        n_total = n_la + self.stage_spec.teammate_count + self.stage_spec.enemy_count
        spawns = []
        random.shuffle(empty_spots)
        for _ in range(n_total):
            best_spot = empty_spots[0]
            if spawns:
                candidates = empty_spots[:20]
                max_min_dist = -1
                for cand in candidates:
                    min_dist = min(math.hypot(cand[0] - s[0], cand[1] - s[1]) for s in spawns)
                    if min_dist > max_min_dist:
                        max_min_dist = min_dist
                        best_spot = cand
            spawns.append(best_spot)
            empty_spots.remove(best_spot)

        idx = 0

        # Learning agents
        self.learning_agents = []
        for i in range(n_la):
            # team_id 由 stage 決定
            if self.stage_id == 5:
                tid = 0 if i == 0 else 1
            elif self.stage_id == 6:
                tid = 0 if i < 2 else 1
            else:
                tid = 0
            color = (0, 140, 255) if tid == 0 else (255, 80, 80)
            team_str = "ai" if i == 0 else f"ai_{i}"
            a = Agent(spawns[idx][0], spawns[idx][1], color, team_str, bot_type="learning")
            a.team_id = tid
            # 出生武器：PISTOL
            a.weapon_slots = [PISTOL]
            a.active_slot = 0
            a.ammo = PISTOL.mag_size
            a.max_ammo = PISTOL.mag_size
            a.reload_delay = PISTOL.reload_frames
            # 身體參數 log-uniform 採樣
            spec = self.stage_spec
            a.body_speed_mult = _sample_log_uniform(*spec.body_speed_range)
            a.body_rot_mult   = _sample_log_uniform(*spec.body_rot_range)
            a.speed = a.base_speed * a.body_speed_mult
            a.infinite_ammo = spec.infinite_ammo
            idx += 1
            self.learning_agents.append(a)

        # Teammate NPC agents
        self.team_agents = []
        for _ in range(self.stage_spec.teammate_count):
            a = Agent(spawns[idx][0], spawns[idx][1], (0, 220, 140), "ally", bot_type="wander")
            a.team_id = 0
            idx += 1
            self.team_agents.append(a)

        # Enemy agents
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
                spawns[idx][0], spawns[idx][1],
                (255, 80, 80), f"enemy_{i}",
                bot_type=bot_type,
                infinite_ammo=(bot_type != "self_play"),
            )
            e.team_id = 1
            e.max_hp = self.stage_spec.enemy_hp
            e.hp = self.stage_spec.enemy_hp
            idx += 1
            self.enemy_agents.append(e)

        self.all_agents = self.learning_agents + self.team_agents + self.enemy_agents
        self.projectiles = []
        self.grenades_list = []
        self.sound_waves = []
        self.frame_count = 0
        self._last_info = {}

        # 散布地面道具
        self.ground_items = []
        n_items = (n_la + self.stage_spec.enemy_count) * 2
        if empty_spots:
            for _ in range(n_items):
                spot = random.choice(empty_spots)
                roll = random.random()
                if roll < _ITEM_TYPE_WEIGHTS[0]:
                    # 武器
                    wp = random.choices(WEAPON_TYPES, weights=_WEAPON_WEIGHTS, k=1)[0]
                    self.ground_items.append(GroundItem(float(spot[0]), float(spot[1]), "weapon", weapon_spec=wp))
                elif roll < _ITEM_TYPE_WEIGHTS[0] + _ITEM_TYPE_WEIGHTS[1]:
                    self.ground_items.append(GroundItem(float(spot[0]), float(spot[1]), "medkit"))
                else:
                    self.ground_items.append(GroundItem(float(spot[0]), float(spot[1]), "grenade"))

        # 毒圈初始化
        if self.stage_spec.has_poison_zone:
            self.poison_cx = (grid_cols * TILE_SIZE) / 2.0
            self.poison_cy = (grid_rows * TILE_SIZE) / 2.0
            self.poison_radius_max = _math.hypot(grid_cols * TILE_SIZE, grid_rows * TILE_SIZE) / 2.0
            self.poison_radius     = self.poison_radius_max
            self.poison_radius_min = TILE_SIZE * 3.0
            # 恰好在 max_frames 幀時縮完
            self.poison_shrink_rate = (
                (self.poison_radius_max - self.poison_radius_min) /
                max(1, self.stage_spec.max_frames)
            )
            self.poison_dmg_per_frame = 0.15
        else:
            self.poison_radius = float('inf')
            self.poison_shrink_rate = 0.0
            self.poison_radius_max = float('inf')

        # 回傳 states
        states = [self._get_local_view(a) for a in self.learning_agents]
        if self.n_learning_agents == 1:
            return states[0]
        return states

    # ── 狀態查詢 ──

    def get_state(self):
        """向後相容，回傳第一個 learning agent 的 state"""
        return self._get_local_view(self.ai_agent)

    def get_all_states(self) -> list:
        return [self._get_local_view(a) for a in self.learning_agents]

    # ── 碰撞與地形 ──

    def is_wall(self, x, y):
        c = int(x // TILE_SIZE)
        r = int(y // TILE_SIZE)
        if 0 <= c < self.grid_cols and 0 <= r < self.grid_rows:
            return self.grid[r, c] == 1
        return True

    def has_line_of_sight(self, x1, y1, x2, y2):
        return njit_has_line_of_sight(
            float(x1), float(y1), float(x2), float(y2),
            self.grid_np, float(TILE_SIZE), self.grid_cols, self.grid_rows,
        )

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

    # ── 雙線性插值 ──

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

    # ═══════════════════════════════════════════════════════
    #  觀測生成（6ch + 22 scalars）
    # ═══════════════════════════════════════════════════════

    def _get_local_view(self, agent) -> Tuple[np.ndarray, np.ndarray]:
        rad = math.radians(agent.angle)
        fwd_x, fwd_y = math.cos(rad), math.sin(rad)
        rgt_x, rgt_y = math.cos(rad + math.pi / 2), math.sin(rad + math.pi / 2)
        ax, ay = agent.x, agent.y

        # 若持狙擊槍，使用 sniper FOV
        wp = agent.active_weapon
        use_sniper = wp is not None and wp.is_sniper

        view = np.zeros((NUM_CHANNELS, VIEW_SIZE, VIEW_SIZE), dtype=np.float32)

        # Ch0: 地形（使用專用函式避免 Numba 重編譯）
        if use_sniper:
            view[0] = njit_compute_fov_sniper(
                float(ax), float(ay),
                float(fwd_x), float(fwd_y),
                float(rgt_x), float(rgt_y),
                self.grid_np,
                float(TILE_SIZE), self.grid_cols, self.grid_rows, VIEW_SIZE,
            )
        else:
            view[0] = njit_compute_fov_standard(
                float(ax), float(ay),
                float(fwd_x), float(fwd_y),
                float(rgt_x), float(rgt_y),
                self.grid_np,
                float(TILE_SIZE), self.grid_cols, self.grid_rows, VIEW_SIZE,
            )

        # Ch1: 敵人雷達（team_id 不同、在 FOV + LOS）
        for other in self.all_agents:
            if not other.alive() or other is agent or other.team_id == agent.team_id:
                continue
            dx = other.x - ax
            dy = other.y - ay
            ft = (dx * fwd_x + dy * fwd_y) / TILE_SIZE
            rt = (dx * rgt_x + dy * rgt_y) / TILE_SIZE
            dt = math.hypot(ft, rt)
            ang = math.degrees(math.atan2(rt, ft)) if dt > 0 else 0.0
            if dt <= VIEW_RANGE and abs(ang) <= HALF_FOV and self.has_line_of_sight(ax, ay, other.x, other.y):
                dr = (ft + rt) / 1.41421356
                dc = (ft - rt) / 1.41421356
                val = other.hp / 200.0
                self._inject_value(view[1], VIEW_CENTER + dr, VIEW_CENTER + dc, val)

        # Ch2: 隊友雷達（team_id 相同、全域可見）
        for other in self.all_agents:
            if not other.alive() or other is agent or other.team_id != agent.team_id:
                continue
            dx = other.x - ax
            dy = other.y - ay
            mft = (dx * fwd_x + dy * fwd_y) / TILE_SIZE
            mrt = (dx * rgt_x + dy * rgt_y) / TILE_SIZE
            dr = (mft + mrt) / 1.41421356
            dc = (mft - mrt) / 1.41421356
            r_f = np.clip(VIEW_CENTER + dr, 0.0, VIEW_SIZE - 1.001)
            c_f = np.clip(VIEW_CENTER + dc, 0.0, VIEW_SIZE - 1.001)
            val = other.hp / 200.0
            self._inject_value(view[2], r_f, c_f, val)

        # Ch3: 威脅/彈道熱力圖
        for p in self.projectiles:
            dx = p.x - ax
            dy = p.y - ay
            ft = (dx * fwd_x + dy * fwd_y) / TILE_SIZE
            rt = (dx * rgt_x + dy * rgt_y) / TILE_SIZE
            dt = math.hypot(ft, rt)
            ang = math.degrees(math.atan2(rt, ft)) if dt > 0 else 0.0
            if dt <= VIEW_RANGE and abs(ang) <= HALF_FOV and self.has_line_of_sight(ax, ay, p.x, p.y):
                if p.owner.team_id != agent.team_id:
                    hv = p.heatmap_value if hasattr(p, 'heatmap_value') else 0.5
                    val = hv
                else:
                    val = -0.2
                dr = (ft + rt) / 1.41421356
                dc = (ft - rt) / 1.41421356
                self._inject_value(view[3], VIEW_CENTER + dr, VIEW_CENTER + dc, val)

        # 手榴彈也加到威脅通道
        for g in self.grenades_list:
            if g.exploded:
                continue
            dx = g.x - ax
            dy = g.y - ay
            ft = (dx * fwd_x + dy * fwd_y) / TILE_SIZE
            rt = (dx * rgt_x + dy * rgt_y) / TILE_SIZE
            dt = math.hypot(ft, rt)
            ang = math.degrees(math.atan2(rt, ft)) if dt > 0 else 0.0
            if dt <= VIEW_RANGE and abs(ang) <= HALF_FOV and self.has_line_of_sight(ax, ay, g.x, g.y):
                val = 0.3 + 0.7 * (g.fuse_timer / max(1, g.fuse_frames))
                dr = (ft + rt) / 1.41421356
                dc = (ft - rt) / 1.41421356
                self._inject_value(view[3], VIEW_CENTER + dr, VIEW_CENTER + dc, val)

        # Ch4: 聲音波紋（逐幀掃描避免 frame_skip 跳格）
        view[4] = render_sound_channel(
            ax, ay, fwd_x, fwd_y, rgt_x, rgt_y,
            self.sound_waves,
            current_frame=self.frame_count,
            prev_frame=self._prev_frame,
            view_size=VIEW_SIZE, view_center=VIEW_CENTER, tile_size=TILE_SIZE,
        )

        # Ch5: 安全區（毒圈距離）
        if self.stage_spec.has_poison_zone and self.poison_radius < float('inf'):
            dist_self = _math.hypot(ax - self.poison_cx, ay - self.poison_cy)
            # 1.0=圈心安全, 0.0=剛好在圈邊, 負值=圈外危險
            view[5] = float(np.clip(1.0 - dist_self / max(1.0, self.poison_radius), -1.0, 1.0))
        else:
            view[5] = 1.0

        # ── 22 純量 ──
        # weapon one-hot
        def _weapon_onehot(slot_idx):
            oh = [0.0] * 5
            if slot_idx < len(agent.weapon_slots) and agent.weapon_slots[slot_idx] is not None:
                wpn = agent.weapon_slots[slot_idx]
                try:
                    wi = WEAPON_TYPES.index(wpn)
                except ValueError:
                    wi = 0
                oh[wi] = 1.0
            else:
                oh[4] = 1.0  # empty
            return oh

        w1_oh = _weapon_onehot(0)
        w2_oh = _weapon_onehot(1)
        active_slot_f = float(agent.active_slot)

        cur_wp = agent.active_weapon
        if cur_wp is not None:
            ammo_ratio = agent.ammo / max(1, cur_wp.mag_size)
            reload_ratio = agent.reload_progress / max(1, cur_wp.reload_frames) if agent.reload_progress > 0 else 0.0
        else:
            ammo_ratio = agent.ammo / max(1, agent.max_ammo)
            reload_ratio = 0.0

        heal_ratio = agent.heal_progress / max(1, agent.heal_frames) if agent.heal_progress > 0 else 0.0
        hp_ratio = agent.hp / max(1, agent.max_hp)
        medkit_ratio = agent.medkits / max(1, agent.max_medkits)
        grenade_ratio = agent.grenades / max(1, agent.max_grenades)
        dash_ratio = agent.dash_cd / 160.0
        hit_marker = 1.0 if getattr(agent, "hit_marker_timer", 0) > 0 else 0.0

        # 最近隊友
        alive_mates = [m for m in self.all_agents
                       if not m.truly_dead() and not m.is_downed()
                       and m is not agent and m.team_id == agent.team_id]
        norm_ft = 0.0
        norm_rt = 0.0
        has_ally = 0.0
        if alive_mates:
            has_ally = 1.0
            closest = min(alive_mates, key=lambda m: math.hypot(m.x - ax, m.y - ay))
            dx = closest.x - ax
            dy = closest.y - ay
            cft = (dx * fwd_x + dy * fwd_y) / TILE_SIZE
            crt = (dx * rgt_x + dy * rgt_y) / TILE_SIZE
            norm_ft = float(np.clip(cft / COLS, -1.0, 1.0))
            norm_rt = float(np.clip(crt / ROWS, -1.0, 1.0))

        scalars = np.array(
            w1_oh + w2_oh + [
                active_slot_f, ammo_ratio, reload_ratio, heal_ratio,
                hp_ratio, medkit_ratio, grenade_ratio, dash_ratio,
                hit_marker, norm_ft, norm_rt, has_ally,
            ],
            dtype=np.float32,
        )

        return view, scalars, agent.team_id

    # ═══════════════════════════════════════════════════════
    #  輔助查詢
    # ═══════════════════════════════════════════════════════

    def _alive_enemies(self):
        """敵人中未 truly_dead 的（含倒地中）"""
        return [e for e in self.enemy_agents if not e.truly_dead()]

    def _alive_allies(self):
        """我方中未 truly_dead 的（含倒地中）"""
        return [a for a in self.learning_agents + self.team_agents if not a.truly_dead()]

    # ═══════════════════════════════════════════════════════
    #  核心步進
    # ═══════════════════════════════════════════════════════

    def _apply_learning_action(self, agent, action_12):
        """
        處理一個 learning agent 的 12 維動作，
        回傳 (did_shoot: bool, dash_reward: float)。
        """
        mask = agent.get_action_mask()
        # 遮罩過濾
        act = [action_12[i] if (i < len(action_12) and mask[i]) else 0.0
               for i in range(12)]

        # 打藥讀條與取消
        if act[9] > 0.5:
            if agent.heal_progress == 0:
                agent.start_heal()
            else:
                # 已經在打藥，再次按下則取消並歸還藥包
                agent.cancel_heal()
                
        if agent.heal_progress > 0:
            agent.tick_heal()

        # 換彈讀條（自動：無彈時自動開始）
        if agent.ammo == 0 and agent.reload_progress == 0 and not agent.infinite_ammo:
            agent.start_reload()
        if agent.reload_progress > 0:
            agent.tick_reload()

        # 武器切換
        if act[8] > 0.5:
            agent.switch_weapon()

        # 投擲手榴彈
        if act[10] > 0.5 and agent.grenades > 0:
            agent.grenades -= 1
            rad2 = math.radians(agent.angle)
            gx = agent.x + math.cos(rad2) * (agent.radius + 8)
            gy = agent.y + math.sin(rad2) * (agent.radius + 8)
            self.grenades_list.append(Grenade(gx, gy, agent.angle, owner=agent))

        # 建構 9 維動作供 apply_actions（原有格式）
        # 0=up,1=down,2=left,3=right,4=cw,5=ccw,6=attack,7=focus,8=dash
        actions_9 = [
            act[0], act[1], act[2], act[3],
            act[4], act[5], act[6],
            act[11],  # focus
            act[7],   # dash
        ]

        # 散彈槍特殊開火
        wp = agent.active_weapon
        if wp is not None and wp.is_shotgun and act[6] > 0.5 and agent.attack_cooldown == 0 and (agent.ammo > 0 or agent.infinite_ammo):
            # 不用 apply_actions 的開火，自己處理多彈片
            agent._tick_base()
            # 移動部分仍需執行
            up, dn, lt, rt_a, cw, ccw = act[0], act[1], act[2], act[3], act[4], act[5]
            fwd_in = (1 if up > 0.5 else 0) - (1 if dn > 0.5 else 0)
            right_in = (1 if rt_a > 0.5 else 0) - (1 if lt > 0.5 else 0)
            turn_in = (1 if cw > 0.5 else 0) - (1 if ccw > 0.5 else 0)
            focus = act[11]
            turn_speed = 1.5 if focus > 0.5 else 8.0
            cur_speed = agent.speed * (3 if agent.dash_timer > 0 else 1)

            dash_reward = 0.0
            if act[7] > 0.5 and agent.dash_cd == 0 and agent.dash_timer == 0 and agent.hp > GameConfig.DASH_COST_HP:
                agent.dash_timer = 10
                agent.dash_cd = 160
                agent.hp -= GameConfig.DASH_COST_HP
                dash_reward = GameConfig.DASH_PENALTY
                cur_speed = agent.speed * 3

            agent.angle = (agent.angle + turn_speed * turn_in) % 360
            rad_a = math.radians(agent.angle)
            fx, fy = math.cos(rad_a), math.sin(rad_a)
            rx, ry = math.cos(rad_a + math.pi / 2), math.sin(rad_a + math.pi / 2)
            dx = fx * cur_speed * fwd_in + rx * cur_speed * right_in
            dy = fy * cur_speed * fwd_in + ry * cur_speed * right_in
            agent._regen_tick(fwd_in, right_in)
            self.try_move_agent(agent, dx, dy)
            agent._reload_tick()

            # 發射多彈片
            base_rad = math.radians(agent.angle)
            sx = agent.x + math.cos(base_rad) * (agent.radius + 5)
            sy = agent.y + math.sin(base_rad) * (agent.radius + 5)
            dmg = self.enemy_damage if agent.team.startswith("enemy") else self.bullet_damage
            if wp:
                dmg = wp.damage
            for offset in _SHOTGUN_OFFSETS:
                pellet_angle = (agent.angle + offset) % 360
                self.projectiles.append(
                    Projectile(sx, sy, pellet_angle, owner=agent, damage=dmg, weapon_spec=wp)
                )
            agent.attack_cooldown = wp.fire_cooldown
            if not agent.infinite_ammo:
                agent.ammo -= 1
            self.sound_waves.append(create_gunshot_wave(agent.x, agent.y, self.frame_count))
            return True, dash_reward

        # 一般開火（含狙擊、步槍、手槍）
        did_shoot, dash_reward = agent.apply_actions(actions_9, self)
        if did_shoot:
            self.sound_waves.append(create_gunshot_wave(agent.x, agent.y, self.frame_count))

        return did_shoot, dash_reward

    def _single_step(self, ai_actions_list, enemy_ai_action=None):
        self._prev_frame = self.frame_count
        self.frame_count += 1

        # 填入 comm_in
        for i, agent in enumerate(self.learning_agents):
            agent.comm_in = [
                getattr(other, 'last_comm_out', np.zeros(4, dtype=np.float32))
                for j, other in enumerate(self.learning_agents)
                if j != i and other.team_id == agent.team_id and other.alive()
            ]

        # 從 stage_spec 讀取本階段的動作誤差比例
        _move_noise_pct = float(getattr(self.stage_spec, "move_noise_pct", 0.0))
        _rot_noise_pct = float(getattr(self.stage_spec, "rotation_noise_pct", 0.0))

        rewards = [0.0] * len(self.learning_agents)

        # Learning agents 執行動作
        for i, agent in enumerate(self.learning_agents):
            if agent.truly_dead():
                continue
            action_12 = ai_actions_list[i] if i < len(ai_actions_list) else [0.0] * 12
            if agent.is_downed():
                # 倒地中只會執行移動，不處理戰鬥動作
                mask = agent.get_action_mask()
                act = [action_12[k] if mask[k] else 0.0 for k in range(12)]
                rad_a = math.radians(agent.angle)
                fx, fy = math.cos(rad_a), math.sin(rad_a)
                rx, ry = math.cos(rad_a + math.pi / 2), math.sin(rad_a + math.pi / 2)
                fwd_in = (1 if act[0] > 0.5 else 0) - (1 if act[1] > 0.5 else 0)
                right_in = (1 if act[3] > 0.5 else 0) - (1 if act[2] > 0.5 else 0)
                dx = fx * agent.speed * agent.downed_speed_ratio * fwd_in
                dy = fy * agent.speed * agent.downed_speed_ratio * fwd_in
                dx += rx * agent.speed * agent.downed_speed_ratio * right_in
                dy += ry * agent.speed * agent.downed_speed_ratio * right_in
                self.try_move_agent(agent, dx, dy)
                continue
            did_shoot, dash_reward = self._apply_learning_action(agent, action_12)
            if did_shoot:
                rewards[i] -= 0.02  # SHOOT_PENALTY
            rewards[i] += dash_reward

        # Teammate NPC 動作
        for mate in self.team_agents:
            if not mate.truly_dead():
                mate.apply_actions(teammate_actions(self, mate), self)

        # Enemy NPC 動作
        for i, enemy in enumerate(self.enemy_agents):
            if enemy.truly_dead():
                continue
            if self.stage_id == 5 and i == 0 and enemy_ai_action is not None:
                enemy.apply_actions(enemy_ai_action, self)
            else:
                enemy.apply_actions(enemy_actions(self, enemy), self)

        # 腳步音效（每 20 幀）
        if self.frame_count % 20 == 0:
            for agent in self.learning_agents + self.enemy_agents:
                if not agent.truly_dead():
                    self.sound_waves.append(create_footstep_wave(agent.x, agent.y, self.frame_count))

        # 倒地救援系統：對每個倒地的 learning agent 検查隊友距離
        for i, agent in enumerate(self.learning_agents):
            if not agent.is_downed():
                continue
            # 尋找同隊且在一格距離內的隊友或小隊
            rescuer_nearby = False
            for other in self.all_agents:
                if other is agent or other.team_id != agent.team_id:
                    continue
                if other.truly_dead() or other.is_downed():
                    continue
                if math.hypot(other.x - agent.x, other.y - agent.y) <= TILE_SIZE:
                    rescuer_nearby = True
                    break
            revived = agent.tick_revive(rescuer_nearby)
            if revived:
                rewards[i] += 3.0   # 被救者獎勵（降低，避免過大）
                # 找最近的同隊友給救援獎勵
                for j, rescuer in enumerate(self.learning_agents):
                    if j == i or rescuer.truly_dead() or rescuer.is_downed():
                        continue
                    if rescuer.team_id == agent.team_id:
                        if math.hypot(rescuer.x - agent.x, rescuer.y - agent.y) <= TILE_SIZE:
                            rewards[j] += 5.0   # 救人者獎勵
                            break

        # 清理過期聲音波紋（使用 frame-based alive 判定）
        self.sound_waves = [w for w in self.sound_waves if w.alive(self.frame_count)]

        # 敵人視野獎勵（per learning agent）
        for i, agent in enumerate(self.learning_agents):
            if agent.truly_dead() or agent.is_downed():
                continue
            frad = math.radians(agent.angle)
            fx, fy = math.cos(frad), math.sin(frad)
            rx, ry = math.cos(frad + math.pi / 2), math.sin(frad + math.pi / 2)
            enemy_in_sight = False
            for other in self.all_agents:
                if not other.alive() or other.team_id == agent.team_id:
                    continue
                dx = other.x - agent.x
                dy = other.y - agent.y
                ft = (dx * fx + dy * fy) / TILE_SIZE
                rt = (dx * rx + dy * ry) / TILE_SIZE
                dt = math.hypot(ft, rt)
                ang = math.degrees(math.atan2(rt, ft)) if dt > 0 else 0.0
                if dt <= VIEW_RANGE and abs(ang) <= HALF_FOV and self.has_line_of_sight(agent.x, agent.y, other.x, other.y):
                    enemy_in_sight = True
                    break
            if enemy_in_sight:
                rewards[i] += 0.005  # RADAR_REWARD

        # 碰撞懲罰
        for i, agent in enumerate(self.learning_agents):
            if agent.truly_dead() or agent.is_downed():
                continue
            for other in self.all_agents:
                if other is agent or not other.alive() or other.team_id == agent.team_id:
                    continue
                if math.hypot(agent.x - other.x, agent.y - other.y) < agent.radius + other.radius + 1:
                    rewards[i] -= 0.005

        # 子彈碰撞
        for p in self.projectiles[:]:
            hit_someone = False
            for ag in self.all_agents:
                if ag.truly_dead() or ag is p.owner:
                    continue
                # 倒地狀態免疫子彈傷害
                if ag.is_downed():
                    continue
                # 防護網邏輯（同隊不傷害）
                if p.owner.team_id == ag.team_id:
                    continue
                if math.hypot(p.x - ag.x, p.y - ag.y) < ag.radius + p.radius:
                    was_downed_before = ag.is_downed()
                    ag.hp -= p.damage
                    # hp 降至 0 且原本不是倒地狀態 → 觸發倒地
                    just_downed = (ag.hp <= 0 and not was_downed_before)
                    if just_downed:
                        ag.enter_downed()
                    # 獎勵 / 懲罰
                    for i, la in enumerate(self.learning_agents):
                        if ag is la:
                            rewards[i] -= 1.0  # DAMAGE_PENALTY
                        if p.owner is la and ag.team_id != la.team_id:
                            # 打到已倒地目標：不給擊中獎勵
                            if not was_downed_before:
                                rewards[i] += 1.0  # HIT_REWARD（只有打到站立目標才給）
                                la.hit_marker_timer = 3
                            # 剛把目標打倒地：擊倒獎勵
                            if just_downed:
                                rewards[i] += 10.0  # DOWN_REWARD
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

        # 手榴彈更新與 AOE
        for g in self.grenades_list[:]:
            if g.exploded:
                self.grenades_list.remove(g)
                continue
            # 撞牆停止
            if self.is_wall(g.x, g.y):
                g.speed = 0.0
            g.update()
            if g.should_explode():
                g.exploded = True
                self.sound_waves.append(create_explosion_wave(g.x, g.y, self.frame_count))
                # AOE 傷害
                for ag in self.all_agents:
                    if ag.truly_dead() or ag.is_downed():
                        continue
                    d = math.hypot(ag.x - g.x, ag.y - g.y)
                    dmg = 0
                    if d <= 60:    # 3×3 格中心
                        dmg = Grenade.CENTER_DAMAGE
                    elif d <= 100:  # 5×5 格外圍
                        dmg = Grenade.OUTER_DAMAGE
                    if dmg > 0:
                        was_downed_before = ag.is_downed()
                        ag.hp -= dmg
                        just_downed = (ag.hp <= 0 and not was_downed_before)
                        if just_downed:
                            ag.enter_downed()
                        # learning agent 獎勵
                        for i, la in enumerate(self.learning_agents):
                            if ag is la:
                                rewards[i] -= 1.0
                            if g.owner is la and ag.team_id != la.team_id:
                                if not was_downed_before:
                                    rewards[i] += 1.0  # HIT_REWARD（非倒地目標）
                                if just_downed:
                                    rewards[i] += 10.0  # DOWN_REWARD
                if g in self.grenades_list:
                    self.grenades_list.remove(g)

        # ── 毒圈更新 ──
        if self.stage_spec.has_poison_zone:
            self.poison_radius = max(
                self.poison_radius_min,
                self.poison_radius - self.poison_shrink_rate
            )
            for i, agent in enumerate(self.learning_agents):
                if agent.truly_dead() or agent.is_downed():
                    continue
                dist = _math.hypot(agent.x - self.poison_cx, agent.y - self.poison_cy)
                if dist > self.poison_radius:
                    overdist_ratio = (dist - self.poison_radius) / (TILE_SIZE * 5)
                    dmg = self.poison_dmg_per_frame * (1.0 + overdist_ratio)
                    agent.hp -= dmg
                    rewards[i] -= 0.05 * (1.0 + overdist_ratio)

        # ── 存活時間指數獎勵（僅毒圈階段）──
        if self.stage_spec.has_poison_zone:
            progress = self.frame_count / max(1, self.stage_spec.max_frames)
            survival_bonus = 0.001 * _math.exp(2.0 * progress)
            for i, agent in enumerate(self.learning_agents):
                if not agent.truly_dead() and not agent.is_downed():
                    rewards[i] += survival_bonus

        # 存活懲罰
        alive_enemy_cnt = len(self._alive_enemies())
        for i, agent in enumerate(self.learning_agents):
            if agent.truly_dead() or agent.is_downed():
                continue
            if self.stage_id in (0, 1, 2, 3, 4):
                rewards[i] -= 0.003 * alive_enemy_cnt
            else:
                rewards[i] -= GameConfig.SURVIVAL_COST

        # 自動拾取
        for agent in self.learning_agents:
            if not agent.truly_dead() and not agent.is_downed():
                try_auto_pickup(agent, self.ground_items)

        # 結束判定
        done = False
        ai_win = False
        ai_lost = False

        # 全員 truly_dead 才算役出（倒地中不算）
        any_la_alive = any(not a.truly_dead() for a in self.learning_agents)
        enemies_alive = alive_enemy_cnt > 0
        allies_alive = len(self._alive_allies()) > 0

        team_reward = 0.0

        if self.stage_id in (0, 1, 2, 3, 4):
            time_out = self.frame_count >= self.stage_spec.max_frames
            if not any_la_alive or not enemies_alive or time_out:
                done = True
                if time_out and enemies_alive:
                    team_reward -= 5.0  # ALIVE_NPC_PENALTY
                if any_la_alive and not enemies_alive:
                    ai_win = True
                    team_reward += 10.0
                elif (not any_la_alive) and enemies_alive:
                    ai_lost = True
                    team_reward -= 5.0
                else:
                    team_reward -= 6.0  # TIE

        elif self.stage_id == 5:
            if not any_la_alive or not enemies_alive or self.frame_count >= self.stage_spec.max_frames:
                done = True
                if any_la_alive and not enemies_alive:
                    ai_win = True
                    team_reward += 10.0
                elif (not any_la_alive) and enemies_alive:
                    ai_lost = True
                    team_reward -= 5.0
                else:
                    team_reward -= 6.0

        elif self.stage_id == 6:
            enemies_team_dead = not enemies_alive
            allies_team_dead = not allies_alive
            if enemies_team_dead or allies_team_dead or self.frame_count >= self.stage_spec.max_frames:
                done = True
                if enemies_team_dead and not allies_team_dead:
                    ai_win = True
                    team_reward += 10.0
                elif allies_team_dead and not enemies_team_dead:
                    ai_lost = True
                    team_reward -= 5.0
                else:
                    team_reward -= 6.0

        # final_reward = 0.6 * individual + 0.4 * team
        final_rewards = [0.6 * rewards[i] + 0.4 * team_reward
                         for i in range(len(self.learning_agents))]

        info = {
            "stage_id": self.stage_id,
            "stage_name": self.stage_spec.name,
            "ai_win": ai_win,
            "ai_lost": ai_lost,
            "ai_alive": any_la_alive,
            "enemy_alive_count": alive_enemy_cnt,
            "ally_alive_count": len([a for a in self.team_agents if a.alive()]),
            "ai_kill_target": ai_win,
            "kill_count": self.stage_spec.enemy_count - alive_enemy_cnt,
            "action_masks": [a.get_action_mask() for a in self.learning_agents],
        }
        self._last_info = info

        states = [self._get_local_view(a) for a in self.learning_agents]
        return states, final_rewards, done, info

    def step(self, ai_actions, enemy_ai_action=None, frame_skip=1):
        """
        ai_actions: List[List[float]] 長度 n_learning_agents，每個長度 12
                    向後相容：單一 List[float]（長度 12）包裝成 [ai_actions]
        """
        # 向後相容：單一動作包裝成 list
        if ai_actions and not isinstance(ai_actions[0], (list, tuple)):
            ai_actions = [ai_actions]

        total_rewards = [0.0] * len(self.learning_agents)
        done = False
        states = None
        info = {}
        for _ in range(frame_skip):
            if done:
                break
            states, rews, done, info = self._single_step(ai_actions, enemy_ai_action=enemy_ai_action)
            for i in range(len(total_rewards)):
                total_rewards[i] += rews[i]

        # 向後相容：n_learning_agents==1 時維持原格式
        if self.n_learning_agents == 1:
            return states[0], total_rewards[0], done, info
        return states, total_rewards, done, info

    # ═══════════════════════════════════════════════════════
    #  渲染
    # ═══════════════════════════════════════════════════════

    def _draw_agent(self, a):
        if a.is_downed():
            # 倒地：縮小的半透明灰色圓圈
            downed_color = (130, 130, 130)
            pygame.draw.circle(self.screen, downed_color, (int(a.x), int(a.y)), a.radius // 2)
            # 救援進度條
            if a.revive_progress > 0:
                bar_w = a.radius * 2
                fill = int(bar_w * a.revive_progress / a.revive_frames)
                bx = int(a.x) - a.radius
                by = int(a.y) + a.radius // 2 + 4
                pygame.draw.rect(self.screen, (60, 60, 60), (bx, by, bar_w, 4))
                pygame.draw.rect(self.screen, (50, 220, 50), (bx, by, fill, 4))
        else:
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

        pts = [(a.x, a.y)]
        steps = 60
        for i in range(steps + 1):
            deg_rel = -HALF_FOV + (FOV_DEGREES * i / steps)
            rad_rel = math.radians(deg_rel)
            cos_rel = math.cos(rad_rel)
            sin_rel = math.sin(rad_rel)
            ft_val = view_r * cos_rel
            rt_val = view_r * sin_rel
            wx = a.x + (fwd_x * ft_val + rgt_x * rt_val) * TILE_SIZE
            wy = a.y + (fwd_y * ft_val + rgt_y * rt_val) * TILE_SIZE
            pts.append((wx, wy))

        pts.append((a.x, a.y))

        if len(pts) > 2:
            pygame.draw.lines(self.screen, (80, 220, 255), True, pts, 1)

    def render(self, info=""):
        if not self.render_mode:
            return
        self.screen.fill((20, 20, 30))

        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                if self.grid[r, c] == 1:
                    pygame.draw.rect(self.screen, (100, 100, 120),
                                     (c * TILE_SIZE, r * TILE_SIZE, TILE_SIZE, TILE_SIZE))

        for p in self.projectiles:
            p.draw(self.screen)

        for a in self.all_agents:
            if a.alive():
                self._draw_agent(a)

        self._draw_fov(self.ai_agent)

        ai = self.ai_agent
        line1 = f"Stage {self.stage_id} {self.stage_spec.name} | AI HP:{ai.hp} Ammo:{ai.ammo}"
        self.screen.blit(self.font.render(line1, True, (255, 255, 255)), (10, 8))

        line2 = f"EnemyAlive:{len(self._alive_enemies())} AllyAlive:{len([a for a in self.team_agents if a.alive()])}"
        self.screen.blit(self.font.render(line2, True, (255, 255, 255)), (10, 30))

        if info:
            self.screen.blit(self.font.render(info, True, (255, 220, 50)), (10, HEIGHT - 28))

        pygame.display.flip()
        self.clock.tick(FPS)

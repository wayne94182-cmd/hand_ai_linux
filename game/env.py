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
NUM_CHANNELS = 10
NUM_SCALARS = 25


# 道具生成權重
_WEAPON_WEIGHTS = [0.2, 0.35, 0.30, 0.15]       # PISTOL, RIFLE, SHOTGUN, SNIPER
_ITEM_TYPE_WEIGHTS = [0.30, 0.15, 0.15, 0.40] # weapon, medkit, grenade, ammo

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
                cum = 0.0
                item = None
                for weight, factory in [
                    (_ITEM_TYPE_WEIGHTS[0], lambda s=spot: GroundItem(float(s[0]), float(s[1]), "weapon", weapon_spec=random.choices(WEAPON_TYPES, weights=_WEAPON_WEIGHTS, k=1)[0])),
                    (_ITEM_TYPE_WEIGHTS[1], lambda s=spot: GroundItem(float(s[0]), float(s[1]), "medkit")),
                    (_ITEM_TYPE_WEIGHTS[2], lambda s=spot: GroundItem(float(s[0]), float(s[1]), "grenade")),
                    (_ITEM_TYPE_WEIGHTS[3], lambda s=spot: GroundItem(float(s[0]), float(s[1]), "ammo")),
                ]:
                    cum += weight
                    if roll < cum:
                        item = factory()
                        break
                if item:
                    self.ground_items.append(item)

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
        use_sniper = wp is not None and getattr(wp, 'name', '') == 'sniper'

        if use_sniper:
            from game.fov import SNIPER_VIEW_RANGE, SNIPER_HALF_FOV, SNIPER_TILE_SIZE
            cur_view_range = float(SNIPER_VIEW_RANGE)
            cur_half_fov = float(SNIPER_HALF_FOV)
            cur_tile_size = float(SNIPER_TILE_SIZE)
        else:
            cur_view_range = float(VIEW_RANGE)
            cur_half_fov = float(HALF_FOV)
            cur_tile_size = float(TILE_SIZE)

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
            ft = (dx * fwd_x + dy * fwd_y) / cur_tile_size
            rt = (dx * rgt_x + dy * rgt_y) / cur_tile_size
            dt = math.hypot(ft, rt)
            ang = math.degrees(math.atan2(rt, ft)) if dt > 0 else 0.0
            if dt <= cur_view_range and abs(ang) <= cur_half_fov and self.has_line_of_sight(ax, ay, other.x, other.y):
                dr = (ft + rt) / 1.41421356
                dc = (ft - rt) / 1.41421356
                # 倒地單位使用 -1.0，讓神經網路能清楚區分「倒地」與「殘血」
                val = -1.0 if other.is_downed() else other.hp / 200.0
                self._inject_value(view[1], VIEW_CENTER + dr, VIEW_CENTER + dc, val)

        # Ch2: 隊友雷達（team_id 相同、全域可見）
        for other in self.all_agents:
            if not other.alive() or other is agent or other.team_id != agent.team_id:
                continue
            dx = other.x - ax
            dy = other.y - ay
            mft = (dx * fwd_x + dy * fwd_y) / cur_tile_size
            mrt = (dx * rgt_x + dy * rgt_y) / cur_tile_size
            dr = (mft + mrt) / 1.41421356
            dc = (mft - mrt) / 1.41421356
            r_f = np.clip(VIEW_CENTER + dr, 0.0, VIEW_SIZE - 1.001)
            c_f = np.clip(VIEW_CENTER + dc, 0.0, VIEW_SIZE - 1.001)
            # 倒地的隊友使用 -1.0，讓神經網路知道需要去救援
            val = -1.0 if other.is_downed() else other.hp / 200.0
            self._inject_value(view[2], r_f, c_f, val)

        # Ch3: 威脅/彈道熱力圖
        for p in self.projectiles:
            dx = p.x - ax
            dy = p.y - ay
            ft = (dx * fwd_x + dy * fwd_y) / cur_tile_size
            rt = (dx * rgt_x + dy * rgt_y) / cur_tile_size
            dt = math.hypot(ft, rt)
            ang = math.degrees(math.atan2(rt, ft)) if dt > 0 else 0.0
            if dt <= cur_view_range and abs(ang) <= cur_half_fov and self.has_line_of_sight(ax, ay, p.x, p.y):
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
            ft = (dx * fwd_x + dy * fwd_y) / cur_tile_size
            rt = (dx * rgt_x + dy * rgt_y) / cur_tile_size
            dt = math.hypot(ft, rt)
            ang = math.degrees(math.atan2(rt, ft)) if dt > 0 else 0.0
            if dt <= cur_view_range and abs(ang) <= cur_half_fov and self.has_line_of_sight(ax, ay, g.x, g.y):
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

        # Ch6: 武器, Ch7: 醫療包, Ch8: 手榴彈, Ch9: 彈藥（地面道具雷達）
        for item in self.ground_items:
            dx = item.x - ax
            dy = item.y - ay
            ft = (dx * fwd_x + dy * fwd_y) / cur_tile_size
            rt = (dx * rgt_x + dy * rgt_y) / cur_tile_size
            dt = math.hypot(ft, rt)
            ang = math.degrees(math.atan2(rt, ft)) if dt > 0 else 0.0
            if dt <= cur_view_range and abs(ang) <= cur_half_fov and self.has_line_of_sight(ax, ay, item.x, item.y):
                dr = (ft + rt) / 1.41421356
                dc = (ft - rt) / 1.41421356
                if item.item_type == "weapon":
                    self._inject_value(view[6], VIEW_CENTER + dr, VIEW_CENTER + dc, 1.0)
                elif item.item_type == "medkit":
                    self._inject_value(view[7], VIEW_CENTER + dr, VIEW_CENTER + dc, 1.0)
                elif item.item_type == "grenade":
                    self._inject_value(view[8], VIEW_CENTER + dr, VIEW_CENTER + dc, 1.0)
                elif item.item_type == "ammo":
                    self._inject_value(view[9], VIEW_CENTER + dr, VIEW_CENTER + dc, 1.0)

        # ── 24 純量 ──
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

        # 新增 2 個純量：倒地狀態 Flag & 救援進度
        is_downed_f = 1.0 if agent.is_downed() else 0.0
        revive_ratio = agent.revive_progress / max(1, agent.revive_frames) if agent.is_downed() else 0.0

        scalars = np.array(
            w1_oh + w2_oh + [
                active_slot_f, ammo_ratio, reload_ratio, heal_ratio,
                hp_ratio, medkit_ratio, grenade_ratio, dash_ratio,
                hit_marker, norm_ft, norm_rt, has_ally,
                is_downed_f, revive_ratio,
                agent.ammo_boxes / max(1, agent.max_ammo_boxes),  # 第 25 個純量：備彈比例
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

    def _apply_learning_action(self, agent, action_16):
        """
        處理一個 learning agent 的 16 維動作，
        回傳 (did_shoot: bool, dash_reward: float)。
        """
        mask = agent.get_action_mask()
        # 遮罩過濾
        act = [action_16[i] if (i < len(action_16) and mask[i]) else 0.0
               for i in range(16)]

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
        if agent.ammo == 0 and agent.reload_progress == 0:
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

        # 開火與移動（含狙擊、步槍、手槍、散彈槍）
        did_shoot, dash_reward = agent.apply_actions(actions_9, self)
        if did_shoot:
            self.sound_waves.append(create_gunshot_wave(agent.x, agent.y, self.frame_count))

        # 丟棄物資（動作 12~15）
        drop_rad = math.radians(agent.angle)
        drop_x = agent.x + math.cos(drop_rad) * 45
        drop_y = agent.y + math.sin(drop_rad) * 45

        if act[12] > 0.5 and len(agent.weapon_slots) > 1:
            wp_to_drop = agent.weapon_slots.pop(agent.active_slot)
            self.ground_items.append(GroundItem(drop_x, drop_y, "weapon", weapon_spec=wp_to_drop))
            agent.active_slot = 0
            if agent.weapon_slots:
                new_wp = agent.weapon_slots[0]
                agent.max_ammo = new_wp.mag_size
                agent.reload_delay = new_wp.reload_frames
                agent.ammo = min(agent.ammo, agent.max_ammo)
            agent.reload_progress = 0
        if act[13] > 0.5 and agent.medkits > 0:
            agent.medkits -= 1
            self.ground_items.append(GroundItem(drop_x, drop_y, "medkit"))
        if act[14] > 0.5 and agent.grenades > 0:
            agent.grenades -= 1
            self.ground_items.append(GroundItem(drop_x, drop_y, "grenade"))
        if act[15] > 0.5 and agent.ammo_boxes > 0:
            agent.ammo_boxes -= 1
            self.ground_items.append(GroundItem(drop_x, drop_y, "ammo"))

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
            action_16 = ai_actions_list[i] if i < len(ai_actions_list) else [0.0] * 16
            if agent.is_downed():
                # 倒地中只會執行移動，不處理戰鬥動作
                mask = agent.get_action_mask()
                act = [action_16[k] if mask[k] else 0.0 for k in range(16)]
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
            did_shoot, dash_reward = self._apply_learning_action(agent, action_16)
            if did_shoot:
                rewards[i] -= GameConfig.SHOOT_PENALTY
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
                rewards[i] += GameConfig.BE_REVIVED_REWARD
                # 找最近的同隊友給救援獎勵
                for j, rescuer in enumerate(self.learning_agents):
                    if j == i or rescuer.truly_dead() or rescuer.is_downed():
                        continue
                    if rescuer.team_id == agent.team_id:
                        if math.hypot(rescuer.x - agent.x, rescuer.y - agent.y) <= TILE_SIZE:
                            rewards[j] += GameConfig.REVIVE_REWARD
                            break

        # 清理過期聲音波紋（使用 frame-based alive 判定）
        self.sound_waves = [w for w in self.sound_waves if w.alive(self.frame_count)]

        # 敵人視野與鎖定獎勵（per learning agent）
        for i, agent in enumerate(self.learning_agents):
            if agent.truly_dead() or agent.is_downed():
                continue
            frad = math.radians(agent.angle)
            fx, fy = math.cos(frad), math.sin(frad)
            rx, ry = math.cos(frad + math.pi / 2), math.sin(frad + math.pi / 2)
            enemy_in_sight = False
            best_angle = 180.0

            # Determine if the agent is using a sniper
            wp = agent.active_weapon
            use_sniper = wp is not None and getattr(wp, 'name', '') == 'sniper'

            if use_sniper:
                from game.fov import SNIPER_VIEW_RANGE, SNIPER_HALF_FOV, SNIPER_TILE_SIZE
                cur_view_range = float(SNIPER_VIEW_RANGE)
                cur_half_fov = float(SNIPER_HALF_FOV)
                cur_tile_size = float(SNIPER_TILE_SIZE)
            else:
                cur_view_range = float(VIEW_RANGE)
                cur_half_fov = float(HALF_FOV)
                cur_tile_size = float(TILE_SIZE)

            for other in self.all_agents:
                if not other.alive() or other.team_id == agent.team_id or other.is_downed():
                    continue
                dx = other.x - agent.x
                dy = other.y - agent.y
                ft = (dx * fx + dy * fy) / cur_tile_size
                rt = (dx * rx + dy * ry) / cur_tile_size
                dt = math.hypot(ft, rt)
                ang = math.degrees(math.atan2(rt, ft)) if dt > 0 else 0.0
                if dt <= cur_view_range and abs(ang) <= cur_half_fov and self.has_line_of_sight(agent.x, agent.y, other.x, other.y):
                    enemy_in_sight = True
                    if abs(ang) < best_angle:
                        best_angle = abs(ang)
                    # We continue checking to find the best angle

            if enemy_in_sight:
                rewards[i] += GameConfig.RADAR_REWARD
                if self.stage_id in (0, 1):
                    AIM_TOLERANCE = 15.0  # 目標必須在準星正負 15 度以內
                    REQUIRED_FRAMES = 5   # 必須連續維持 5 個 Frame
                    
                    if best_angle <= AIM_TOLERANCE:
                        agent.aim_frames += 1
                    else:
                        agent.aim_frames = 0
                    
                    if agent.aim_frames >= REQUIRED_FRAMES:
                        rewards[i] += GameConfig.aim_reward
                        agent.aim_frames = 0
            else:
                agent.aim_frames = 0

        # 碰撞懲罰
        for i, agent in enumerate(self.learning_agents):
            if agent.truly_dead() or agent.is_downed():
                continue
            for other in self.all_agents:
                if other is agent or not other.alive() or other.team_id == agent.team_id:
                    continue
                if math.hypot(agent.x - other.x, agent.y - other.y) < agent.radius + other.radius + 1:
                    rewards[i] -= GameConfig.COLLISION_PENALTY

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
                            rewards[i] -= GameConfig.DAMAGE_PENALTY_COEF * p.damage
                            if just_downed:
                                rewards[i] -= GameConfig.BE_DOWNED_PENALTY
                        if p.owner is la and ag.team_id != la.team_id:
                            # 打到已倒地目標：不給擊中獎勵
                            if not was_downed_before:
                                rewards[i] += GameConfig.HIT_REWARD_COEF * p.damage
                                la.hit_marker_timer = 3
                            # 剛把目標打倒地：擊倒獎勵
                            if just_downed:
                                rewards[i] += GameConfig.DOWN_REWARD
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
                                rewards[i] -= GameConfig.DAMAGE_PENALTY_COEF * dmg
                                if just_downed:
                                    rewards[i] -= GameConfig.BE_DOWNED_PENALTY
                            if g.owner is la and ag.team_id != la.team_id:
                                if not was_downed_before:
                                    rewards[i] += GameConfig.HIT_REWARD_COEF * dmg
                                if just_downed:
                                    rewards[i] += GameConfig.DOWN_REWARD
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
                    was_downed_before = agent.is_downed()
                    agent.hp -= dmg
                    just_downed = (agent.hp <= 0 and not was_downed_before)
                    if just_downed:
                        agent.enter_downed()
                    rewards[i] -= 0.05 * (1.0 + overdist_ratio)
                    if just_downed:
                        rewards[i] -= GameConfig.BE_DOWNED_PENALTY

        # ── 存活時間指數獎勵（僅毒圈階段）──
        if self.stage_spec.has_poison_zone:
            progress = self.frame_count / max(1, self.stage_spec.max_frames)
            survival_bonus = 0.001 * _math.exp(2.0 * progress)
            for i, agent in enumerate(self.learning_agents):
                if not agent.truly_dead() and not agent.is_downed():
                    rewards[i] += survival_bonus

        # 存活懲罰：以「尚未倒地」的敵方 NPC 數量為基準
        standing_enemies_cnt = sum(1 for e in self.enemy_agents if not e.truly_dead() and not e.is_downed())
        for i, agent in enumerate(self.learning_agents):
            if agent.truly_dead() or agent.is_downed():
                continue
            if self.stage_id in (0, 1, 2, 3, 4):
                rewards[i] -= GameConfig.NPC_SURVIVAL_COST * standing_enemies_cnt
            else:
                rewards[i] -= GameConfig.SURVIVAL_COST

        # 自動拾取
        for agent in self.learning_agents:
            if not agent.truly_dead() and not agent.is_downed():
                try_auto_pickup(agent, self.ground_items)

        # 隊伍團滅檢查：所有存活成員皆為倒地狀態 => 團隊滅亡，全數轉換為 dead
        teams = {}
        for agent in self.all_agents:
            if not agent.truly_dead():
                teams.setdefault(agent.team_id, []).append(agent)
                
        for team_id, members in teams.items():
            if all(m.is_downed() for m in members):
                for m in members:
                    m.downed = False
                    m.hp = 0
                # 發放擊殺獎勵給不同隊伍的 learning_agents
                for i, la in enumerate(self.learning_agents):
                    if la.team_id != team_id:
                        rewards[i] += GameConfig.NPC_KILL_REWARD * len(members)

        # 結束判定
        done = False
        ai_win = False
        ai_lost = False

        # 全員 truly_dead 才算役出（倒地中不算）
        any_la_alive = any(not a.truly_dead() for a in self.learning_agents)
        alive_enemy_cnt = len(self._alive_enemies())
        enemies_alive = alive_enemy_cnt > 0
        allies_alive = len(self._alive_allies()) > 0

        team_reward = 0.0

        if self.stage_id in (0, 1, 2, 3, 4):
            time_out = self.frame_count >= self.stage_spec.max_frames
            if not any_la_alive or not enemies_alive or time_out:
                done = True
                if time_out and enemies_alive:
                    team_reward -= GameConfig.ALIVE_NPC_PENALTY
                if any_la_alive and not enemies_alive:
                    ai_win = True
                    team_reward += GameConfig.WIN_REWARD
                elif (not any_la_alive) and enemies_alive:
                    ai_lost = True
                    team_reward -= GameConfig.LOSE_PENALTY
                else:
                    team_reward -= GameConfig.TIE_PENALTY

        elif self.stage_id == 5:
            if not any_la_alive or not enemies_alive or self.frame_count >= self.stage_spec.max_frames:
                done = True
                if any_la_alive and not enemies_alive:
                    ai_win = True
                    team_reward += GameConfig.WIN_REWARD
                elif (not any_la_alive) and enemies_alive:
                    ai_lost = True
                    team_reward -= GameConfig.LOSE_PENALTY
                else:
                    team_reward -= GameConfig.TIE_PENALTY

        elif self.stage_id == 6:
            enemies_team_dead = not enemies_alive
            allies_team_dead = not allies_alive
            if enemies_team_dead or allies_team_dead or self.frame_count >= self.stage_spec.max_frames:
                done = True
                if enemies_team_dead and not allies_team_dead:
                    ai_win = True
                    team_reward += GameConfig.WIN_REWARD
                elif allies_team_dead and not enemies_team_dead:
                    ai_lost = True
                    team_reward -= GameConfig.LOSE_PENALTY
                else:
                    team_reward -= GameConfig.TIE_PENALTY

        # final_reward = 0.6 * individual + 0.4 * team
        final_rewards = [GameConfig.INDIVIDUAL_REWARD_WEIGHT * rewards[i] + GameConfig.TEAM_REWARD_WEIGHT * team_reward
                         for i in range(len(self.learning_agents))]

        downed_or_dead_cnt = sum(1 for e in self.enemy_agents if e.is_downed() or e.truly_dead())
        info = {
            "stage_id": self.stage_id,
            "stage_name": self.stage_spec.name,
            "ai_win": ai_win,
            "ai_lost": ai_lost,
            "ai_alive": any_la_alive,
            "enemy_alive_count": alive_enemy_cnt,
            "ally_alive_count": len([a for a in self.team_agents if a.alive()]),
            "ai_kill_target": ai_win,
            "down_count": downed_or_dead_cnt,
            "action_masks": [a.get_action_mask() for a in self.learning_agents],
        }
        self._last_info = info

        states = [self._get_local_view(a) for a in self.learning_agents]
        return states, final_rewards, done, info

    def step(self, ai_actions, enemy_ai_action=None, frame_skip=1):
        """
        ai_actions: List[List[float]] 長度 n_learning_agents，每個長度 16
                    向後相容：單一 List[float]（長度 16）包裝成 [ai_actions]
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
            # 倒地：縮小但保留原本顏色的半透明感，外加明顯的紅色打叉，避免使用者以為模型消失了
            downed_color = (max(0, a.color[0]-100), max(0, a.color[1]-100), max(0, a.color[2]-100))
            pygame.draw.circle(self.screen, downed_color, (int(a.x), int(a.y)), a.radius // 2)
            pygame.draw.circle(self.screen, (255, 50, 50), (int(a.x), int(a.y)), a.radius // 2, 2)
            # 畫個紅叉
            r = a.radius // 2
            pygame.draw.line(self.screen, (255, 50, 50), (int(a.x) - r, int(a.y) - r), (int(a.x) + r, int(a.y) + r), 2)
            pygame.draw.line(self.screen, (255, 50, 50), (int(a.x) - r, int(a.y) + r), (int(a.x) + r, int(a.y) - r), 2)
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
        half_fov_val = HALF_FOV
        fov_degrees_val = FOV_DEGREES
        tile_size_val = float(TILE_SIZE)
        if a.active_weapon and getattr(a.active_weapon, 'name', '') == 'sniper':
            from game.fov import SNIPER_VIEW_RANGE, SNIPER_HALF_FOV, SNIPER_FOV_DEGREES, SNIPER_TILE_SIZE
            view_r = float(SNIPER_VIEW_RANGE)
            half_fov_val = float(SNIPER_HALF_FOV)
            fov_degrees_val = float(SNIPER_FOV_DEGREES)
            tile_size_val = float(SNIPER_TILE_SIZE)

        pts = [(a.x, a.y)]
        steps = 60
        for i in range(steps + 1):
            deg_rel = -half_fov_val + (fov_degrees_val * i / steps)
            rad_rel = math.radians(deg_rel)
            cos_rel = math.cos(rad_rel)
            sin_rel = math.sin(rad_rel)
            ft_val = view_r * cos_rel
            rt_val = view_r * sin_rel
            wx = a.x + (fwd_x * ft_val + rgt_x * rt_val) * tile_size_val
            wy = a.y + (fwd_y * ft_val + rgt_y * rt_val) * tile_size_val
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

"""
game/entities.py — Agent, Projectile, Grenade
從 game/env.py 搬移 Agent / Projectile 並擴充新實體系統。
"""
import math
import random
from typing import Optional

import pygame

from game.config import (
    FPS, NUM_ACTIONS, GameConfig,
)
from game.items import WeaponSpec


# ═══════════════════════════════════════════════════════
#  Agent
# ═══════════════════════════════════════════════════════

class Agent:
    def __init__(self, x, y, color, team, bot_type="wander", infinite_ammo=False):
        self.x = x
        self.y = y
        self.radius = 15
        self.color = color

        # team 可接受 str（舊 API）或 int（新 API）
        if isinstance(team, int):
            self.team_id: int = team
            self.team: str = {0: "ai", 1: "enemy_0"}.get(team, str(team))
        else:
            self.team = team
            if team in ("ai", "ally"):
                self.team_id = 0
            elif team.startswith("enemy"):
                self.team_id = 1
            else:
                self.team_id = -1

        self.bot_type = bot_type
        self.speed = 3
        self.angle = random.randint(0, 359)
        self.hp = 100
        self.max_hp = 100
        self.attack_cooldown = 0
        self.ammo = 5
        self.max_ammo = 5
        self.reload_timer = 0
        self.reload_delay = 180
        self.dash_timer = 0
        self.dash_cd = 0
        self.regen_timer = 0
        self.bot_turn_hold = random.randint(20, 80)
        self.bot_move_hold = random.randint(30, 100)

        self.aim_frames = 0
        self.hit_marker_timer = 0
        self.infinite_ammo = infinite_ammo

        # ── 新增：武器 / 背包 / 讀條 ──
        self.weapon_slots: list = []   # 長度最多 2，元素為 WeaponSpec 或 None
        self.active_slot: int = 0
        self.comm_in: list = []        # 本幀收到的隊友通訊向量列表

        # 背包
        self.medkits: int = 0
        self.max_medkits: int = 5
        self.grenades: int = 0
        self.max_grenades: int = 2

        # 讀條狀態
        self.reload_progress: int = 0  # 當前換彈已累積幀數（0 = 未換彈）
        self.heal_progress: int = 0    # 當前打藥已累積幀數（0 = 未打藥）
        self.heal_frames: int = 90     # 打一包藥需要的幀數
        self.heal_amount: int = 50     # 每包回復 HP

    # ── properties ──

    @property
    def active_weapon(self) -> Optional[WeaponSpec]:
        if not self.weapon_slots:
            return None
        if self.active_slot < len(self.weapon_slots):
            return self.weapon_slots[self.active_slot]
        return None

    # ── 存活 ──

    def alive(self):
        return self.hp > 0

    # ── 基礎 tick ──

    def _tick_base(self):
        if self.dash_cd > 0:
            self.dash_cd -= 1
        if self.dash_timer > 0:
            self.dash_timer -= 1
        if self.attack_cooldown > 0:
            self.attack_cooldown -= 1
        if getattr(self, "hit_marker_timer", 0) > 0:
            self.hit_marker_timer -= 1

    def _reload_tick(self):
        if not self.infinite_ammo and self.ammo == 0:
            self.reload_timer += 1
            if self.reload_timer >= self.reload_delay:
                self.ammo = self.max_ammo
                self.reload_timer = 0

    def _regen_tick(self, fwd_in, right_in):
        if fwd_in == 0 and right_in == 0:
            self.regen_timer += 1
            if self.regen_timer >= 240:
                self.hp = min(self.max_hp, self.hp + GameConfig.REGEN_AMOUNT)
                self.regen_timer = 0
        else:
            self.regen_timer = 0

    # ── 動作執行（原有邏輯完全保留）──

    # apply_actions
    #     move_noise_pct    : 移動速度誤差比例，0.05 = ±5%。
    #                         實際效果：cur_speed × uniform(1-pct, 1+pct)
    #     rotation_noise_pct: 轉向速度誤差比例，0.05 = ±5%。
    #                         實際效果：turn_speed × uniform(1-pct, 1+pct)
    #
    def apply_actions(self, actions, env, move_noise_pct: float = 0.0, rotation_noise_pct: float = 0.0):
        """apply_actions
        move_noise_pct    : 移動速度誤差比例，0.05 = ±5%。
                            實際效果：cur_speed × uniform(1-pct, 1+pct)
        rotation_noise_pct: 轉向速度誤差比例，0.05 = ±5%。
                            實際效果：turn_speed × uniform(1-pct, 1+pct)
        """
        if not self.alive():
            return False, 0.0

        self._tick_base()

        up, dn, lt, rt, cw, ccw, atk, focus, dash_btn = actions
        fwd_in = (1 if up > 0.5 else 0) - (1 if dn > 0.5 else 0)
        right_in = (1 if rt > 0.5 else 0) - (1 if lt > 0.5 else 0)
        turn_in = (1 if cw > 0.5 else 0) - (1 if ccw > 0.5 else 0)

        turn_speed = 1.5 if focus > 0.5 else 8.0
        cur_speed = self.speed * (3 if self.dash_timer > 0 else 1)

        dash_reward = 0.0
        if dash_btn > 0.5 and self.dash_cd == 0 and self.dash_timer == 0 and self.hp > GameConfig.DASH_COST_HP:
            self.dash_timer = 10
            self.dash_cd = 160
            self.hp -= GameConfig.DASH_COST_HP
            dash_reward = GameConfig.DASH_PENALTY
            cur_speed = self.speed * 3

        # --- 轉向誤差（比例）：turn_speed × uniform(1-pct, 1+pct) ---
        if rotation_noise_pct > 0 and turn_in != 0:
            rot_scale = random.uniform(1.0 - rotation_noise_pct, 1.0 + rotation_noise_pct)
            effective_turn = turn_speed * rot_scale
        else:
            effective_turn = turn_speed
        self.angle = (self.angle + effective_turn * turn_in) % 360
        rad = math.radians(self.angle)
        fx, fy = math.cos(rad), math.sin(rad)
        rx, ry = math.cos(rad + math.pi / 2), math.sin(rad + math.pi / 2)

        # --- 移動誤差（比例）：cur_speed × uniform(1-pct, 1+pct) ---
        if move_noise_pct > 0 and (fwd_in != 0 or right_in != 0):
            move_scale = random.uniform(1.0 - move_noise_pct, 1.0 + move_noise_pct)
            effective_speed = cur_speed * move_scale
        else:
            effective_speed = cur_speed
        dx = fx * effective_speed * fwd_in + rx * effective_speed * right_in
        dy = fy * effective_speed * fwd_in + ry * effective_speed * right_in

        self._regen_tick(fwd_in, right_in)
        env.try_move_agent(self, dx, dy)
        self._reload_tick()

        can_shoot = atk > 0.5 and self.attack_cooldown == 0 and (self.ammo > 0 or self.infinite_ammo)
        did_shoot = False
        if can_shoot:
            rad2 = math.radians(self.angle)
            sx = self.x + math.cos(rad2) * (self.radius + 5)
            sy = self.y + math.sin(rad2) * (self.radius + 5)
            env.projectiles.append(Projectile(sx, sy, self.angle, owner=self, damage=env.enemy_damage if self.team.startswith("enemy") else env.bullet_damage))
            if self.infinite_ammo:
                fire_rate = float(getattr(env.stage_spec, "enemy_fire_rate", 0.0))
                self.attack_cooldown = max(1, int(FPS / fire_rate)) if fire_rate > 0 else 15
            else:
                self.attack_cooldown = 15
                self.ammo -= 1
            did_shoot = True

        return did_shoot, dash_reward

    # ── 新增方法：Action Masking ──

    def get_action_mask(self) -> list:
        """
        回傳長度 12 的 bool list，True = 此動作允許執行。
        索引對應：
          0=up, 1=down, 2=left, 3=right,
          4=cw, 5=ccw, 6=attack, 7=dash,
          8=switch_weapon, 9=use_medkit,
          10=throw_grenade, 11=focus
        """
        mask = [True] * 12

        # 換彈中：禁止 attack(6), switch_weapon(8)
        if self.reload_progress > 0:
            mask[6] = False
            mask[8] = False

        # 打藥中：只允許轉向 (cw=4, ccw=5)
        if self.heal_progress > 0:
            for i in range(12):
                if i not in (4, 5):
                    mask[i] = False

        # 沒有武器：禁止 attack(6)
        if not self.weapon_slots or self.active_weapon is None:
            mask[6] = False

        # 沒有彈藥且非無限彈藥：禁止 attack(6)
        if self.ammo <= 0 and not self.infinite_ammo:
            mask[6] = False

        # 沒有藥 或 HP 已滿：禁止 use_medkit(9)
        if self.medkits <= 0 or self.hp >= self.max_hp:
            mask[9] = False

        # 沒有手榴彈：禁止 throw_grenade(10)
        if self.grenades <= 0:
            mask[10] = False

        # 只有一把武器或沒有武器：禁止 switch_weapon(8)
        if len(self.weapon_slots) < 2:
            mask[8] = False

        return mask

    # ── 新增方法：武器切換 ──

    def switch_weapon(self):
        """切換到另一個武器槽，若另一槽為空則無效"""
        if len(self.weapon_slots) < 2:
            return
        new_slot = 1 - self.active_slot
        if new_slot < len(self.weapon_slots) and self.weapon_slots[new_slot] is not None:
            self.active_slot = new_slot
            wp = self.weapon_slots[new_slot]
            # 切槍時同步彈藥數據
            self.max_ammo = wp.mag_size
            self.reload_delay = wp.reload_frames
            # 重置換彈進度
            self.reload_progress = 0

    # ── 新增方法：換彈讀條 ──

    def start_reload(self):
        """開始換彈，設定 reload_progress = 1"""
        if self.reload_progress > 0:
            return  # 已經在換彈中
        if self.ammo >= self.max_ammo:
            return  # 滿彈匣不需要換
        self.reload_progress = 1

    def tick_reload(self) -> bool:
        """每幀呼叫，回傳 True 代表換彈完成"""
        if self.reload_progress <= 0:
            return False
        wp = self.active_weapon
        required = wp.reload_frames if wp else self.reload_delay
        self.reload_progress += 1
        if self.reload_progress >= required:
            self.ammo = self.max_ammo
            self.reload_progress = 0
            self.reload_timer = 0
            return True
        return False

    # ── 新增方法：打藥讀條 ──

    def start_heal(self):
        """消耗一個藥包，開始打藥讀條"""
        if self.heal_progress > 0:
            return  # 已經在打藥
        if self.medkits <= 0:
            return
        if self.hp >= self.max_hp:
            return
        self.medkits -= 1
        self.heal_progress = 1

    def tick_heal(self) -> bool:
        """每幀呼叫，回傳 True 代表打藥完成"""
        if self.heal_progress <= 0:
            return False
        self.heal_progress += 1
        if self.heal_progress >= self.heal_frames:
            self.hp = min(self.max_hp, self.hp + self.heal_amount)
            self.heal_progress = 0
            return True
        return False


# ═══════════════════════════════════════════════════════
#  Projectile
# ═══════════════════════════════════════════════════════

class Projectile:
    def __init__(self, x, y, angle, owner, damage, weapon_spec: Optional[WeaponSpec] = None):
        self.x = x
        self.y = y
        self.angle = angle
        self.owner = owner
        self.damage = damage
        self.weapon_spec = weapon_spec

        # 若有 weapon_spec，使用其參數；否則使用預設值
        if weapon_spec is not None:
            self.speed = weapon_spec.bullet_speed
            self.life = weapon_spec.bullet_life
            self.radius = 5
            self.heatmap_value = weapon_spec.heatmap_value
        else:
            self.speed = 18
            self.life = 22
            self.radius = 5
            self.heatmap_value = 0.5

    def update(self):
        rad = math.radians(self.angle)
        self.x += math.cos(rad) * self.speed
        self.y += math.sin(rad) * self.speed
        self.life -= 1

    def draw(self, screen):
        pygame.draw.circle(screen, (255, 220, 50), (int(self.x), int(self.y)), self.radius)


# ═══════════════════════════════════════════════════════
#  Grenade
# ═══════════════════════════════════════════════════════

class Grenade:
    """手榴彈，延遲引爆"""
    # 爆炸參數（由 env 讀取使用）
    CENTER_DAMAGE = 80   # 3×3 格中心傷害
    OUTER_DAMAGE = 40    # 5×5 格外圍傷害
    CENTER_RANGE = 3     # 中心範圍（格數）
    OUTER_RANGE = 5      # 外圍範圍（格數）

    def __init__(self, x, y, angle, owner):
        self.x = x
        self.y = y
        self.angle = angle
        self.owner = owner
        self.speed = 8.0
        self.radius = 6
        self.fuse_frames = 90        # 1.5 秒後爆炸
        self.fuse_timer = 0
        self.exploded = False

    def update(self):
        """移動 + 倒數引信"""
        if self.exploded:
            return
        # 移動（手榴彈隨時間減速）
        if self.speed > 0:
            rad = math.radians(self.angle)
            self.x += math.cos(rad) * self.speed
            self.y += math.sin(rad) * self.speed
            # 摩擦力減速
            self.speed = max(0.0, self.speed - 0.15)
        # 引信倒數
        self.fuse_timer += 1

    def should_explode(self) -> bool:
        return self.fuse_timer >= self.fuse_frames

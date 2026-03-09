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
        self.speed = GameConfig.AGENT_BASE_SPEED
        self.base_speed = GameConfig.AGENT_BASE_SPEED   # 永久記錄出生速度，供 reset 時乘倍率
        self.body_speed_mult = 1.0
        self.body_rot_mult = 1.0
        self.angle = random.randint(0, 359)
        self.hp = GameConfig.AGENT_MAX_HP
        self.max_hp = GameConfig.AGENT_MAX_HP
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

        # ── 倒地系統 ──
        self.downed: bool = False          # 是否處於倒地狀態
        self.revive_progress: int = 0      # 救援累積幀數
        self.revive_frames: int = 300      # 救援所需幀數（5 秒 × 60 FPS）
        self.downed_speed_ratio: float = 0.2  # 倒地時移動速度倍率
        self.downed_timer: int = 0         # 倒地已經過的幀數
        self.downed_timeout: int = 600     # 10 秒後若未被救則真正死亡

    # ── properties ──

    @property
    def active_weapon(self) -> Optional[WeaponSpec]:
        if not self.weapon_slots:
            return None
        if self.active_slot < len(self.weapon_slots):
            return self.weapon_slots[self.active_slot]
        return None

    # ── 存活 ──

    def alive(self) -> bool:
        """倒地或正常存活都回傳 True，只有 truly_dead() 為 True 才算徹底死亡。"""
        return not self.truly_dead()

    def truly_dead(self) -> bool:
        """真正死亡：hp <= 0 且不處於倒地救援期（倒地資格用盡後 hp 仍 <= 0）。"""
        return self.hp <= 0 and not self.downed

    def is_downed(self) -> bool:
        return self.downed

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

    def _regen_tick(self, fwd_in=0, right_in=0):
        pass  # 回血機制已停用

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
        if self.truly_dead():
            return False, 0.0

        self._tick_base()

        up, dn, lt, rt, cw, ccw, atk, focus, dash_btn = actions
        fwd_in = (1 if up > 0.5 else 0) - (1 if dn > 0.5 else 0)
        right_in = (1 if rt > 0.5 else 0) - (1 if lt > 0.5 else 0)
        turn_in = (1 if cw > 0.5 else 0) - (1 if ccw > 0.5 else 0)

        body_rot_mult = getattr(self, 'body_rot_mult', 1.0)
        turn_speed = (1.5 if focus > 0.5 else 8.0) * body_rot_mult
        # 注意：移動速度倍率已在 reset() 時乘入 self.speed，這裡不需再乘
        if self.downed:
            cur_speed = self.speed * self.downed_speed_ratio
        else:
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
            
            wp = self.active_weapon
            dmg = env.enemy_damage if self.team.startswith("enemy") else (wp.damage if wp else env.bullet_damage)
            
            if wp is not None and getattr(wp, 'is_shotgun', False):
                for offset in wp.pellet_offsets:
                    pellet_angle = (self.angle + offset) % 360
                    env.projectiles.append(Projectile(sx, sy, pellet_angle, owner=self, damage=dmg, weapon_spec=wp))
            else:
                env.projectiles.append(Projectile(sx, sy, self.angle, owner=self, damage=dmg, weapon_spec=wp))
                
            if wp is not None:
                self.attack_cooldown = wp.fire_cooldown
            elif self.infinite_ammo:
                fire_rate = float(getattr(env.stage_spec, "enemy_fire_rate", 0.0))
                self.attack_cooldown = max(1, int(FPS / fire_rate)) if fire_rate > 0 else 15
            else:
                self.attack_cooldown = 15
                
            if not self.infinite_ammo:
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

        # 倒地中：只允許移動（緩慢爬行），禁止一切戰鬥動作
        if self.downed:
            for i in range(12):
                if i not in (0, 1, 2, 3):   # 只保留 up/down/left/right
                    mask[i] = False
            return mask

        # 換彈中：禁止 attack(6), switch_weapon(8)
        if self.reload_progress > 0:
            mask[6] = False
            mask[8] = False

        # 打藥中：只允許轉向 (cw=4, ccw=5) 與取消打藥 (use_medkit=9)
        if self.heal_progress > 0:
            for i in range(12):
                if i not in (4, 5, 9):
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

    def cancel_heal(self):
        """中斷打藥，並歸還藥包"""
        if self.heal_progress > 0:
            self.heal_progress = 0
            self.medkits += 1

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

    # ── 倒地救援 ──

    def enter_downed(self):
        """進入倒地狀態：hp 鎖在 1，清除各種讀條"""
        self.downed = True
        self.hp = 1
        self.heal_progress = 0
        self.reload_progress = 0
        self.revive_progress = 0
        self.downed_timer = 0

    def tick_revive(self, rescuer_nearby: bool) -> bool:
        """
        每幀呼叫：有隊友靠近時累積救援進度。
        回傳 True 代表救援完成，Agent 已恢復戰力。
        超時未被救則真正死亡。
        """
        if not self.downed:
            return False
        self.downed_timer += 1
        # 超時未被救 → 真正死亡
        if self.downed_timer >= self.downed_timeout:
            self.downed = False
            self.hp = 0
            return False
        if rescuer_nearby:
            self.revive_progress += 1
        if self.revive_progress >= self.revive_frames:
            self.downed = False
            self.revive_progress = 0
            self.downed_timer = 0
            self.hp = 50  # 救起後恢復半血
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
        else:
            self.speed = 18

        # 計算飛出視野所需的最大幀數，避免移動速度過快時無限飛行
        # 預設視野半徑為 VIEW_RANGE (10) * TILE_SIZE (40) = 400
        view_dist = 400.0
        if weapon_spec is not None and getattr(weapon_spec, 'name', '') == 'sniper':
            from game.fov import SNIPER_VIEW_RANGE
            # Sniper 有特殊的視距 (預設 10格但縮放變兩倍實體距離 或 直接更遠)
            # 在 fov.py 中 sniper 視野為 VIEW_RANGE * TILE_SIZE * 2 (或更遠)
            view_dist = 800.0 # 給予兩倍距離

        max_valid_life = int(math.ceil(view_dist / self.speed))

        if weapon_spec is not None:
            # 放寬 sniper 子彈存活限制，確保能飛到目標
            self.life = min(weapon_spec.bullet_life, max_valid_life)
            self.radius = 5
            self.heatmap_value = weapon_spec.heatmap_value
        else:
            self.life = min(22, max_valid_life)
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

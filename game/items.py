from dataclasses import dataclass, field
from typing import Optional
import math


@dataclass
class WeaponSpec:
    name: str
    damage: int
    mag_size: int
    reload_frames: int     # 換彈所需幀數
    fire_cooldown: int     # 開火後冷卻幀數
    bullet_speed: float
    bullet_life: int       # 子彈最大存活幀數
    spread_deg: float      # 散布角度（半角）
    heatmap_value: float   # 在威脅通道的像素亮度 [0,1]
    is_sniper: bool = False
    is_shotgun: bool = False
    pellet_count: int = 1  # 散彈同時發射的彈片數


# 四把武器的精確數值
PISTOL = WeaponSpec(
    name="pistol", damage=10, mag_size=10,
    reload_frames=120, fire_cooldown=20,
    bullet_speed=18.0, bullet_life=22,
    spread_deg=2.0, heatmap_value=0.2
)
RIFLE = WeaponSpec(
    name="rifle", damage=20, mag_size=15,
    reload_frames=150, fire_cooldown=8,
    bullet_speed=22.0, bullet_life=28,
    spread_deg=5.0, heatmap_value=0.5
)
SHOTGUN = WeaponSpec(
    name="shotgun", damage=20, mag_size=5,
    reload_frames=180, fire_cooldown=35,
    bullet_speed=15.0, bullet_life=10,
    spread_deg=0.0, heatmap_value=0.4,
    is_shotgun=True, pellet_count=5
    # 5 發彈片，各自散開 0, ±15, ±30 度
)
SNIPER = WeaponSpec(
    name="sniper", damage=100, mag_size=5,
    reload_frames=210, fire_cooldown=75,
    bullet_speed=40.0, bullet_life=50,
    spread_deg=0.0, heatmap_value=1.0,
    is_sniper=True
)

WEAPON_TYPES = [PISTOL, RIFLE, SHOTGUN, SNIPER]
# one-hot index: pistol=0, rifle=1, shotgun=2, sniper=3, empty=4


@dataclass
class GroundItem:
    """地面上可撿取的道具"""
    x: float
    y: float
    item_type: str          # "weapon", "medkit", "grenade"
    weapon_spec: Optional[WeaponSpec] = None  # item_type=="weapon" 時有效


def try_auto_pickup(agent, ground_items: list, pickup_radius: float = 40.0) -> list:
    """
    對 ground_items 中距離 agent 夠近的道具嘗試自動拾取。

    武器：背包未滿（<2 槽）時直接拾取；
          已滿時若地面武器比手上最差的武器「更好」（依 damage 判斷），
          自動丟棄最差武器，改拾取新武器（rule-based，不需 AI 決策）。
    藥包：medkits < max_medkits 時拾取。
    手榴彈：grenades < max_grenades 時拾取。

    回傳被移除的 ground_items 項目列表。
    """
    picked_up = []

    for item in ground_items[:]:  # iterate over a copy
        dist = math.hypot(agent.x - item.x, agent.y - item.y)
        if dist > pickup_radius:
            continue

        if item.item_type == "weapon" and item.weapon_spec is not None:
            wp = item.weapon_spec
            if len(agent.weapon_slots) < 2:
                # 背包未滿，直接拾取
                agent.weapon_slots.append(wp)
                # 同步 ammo / reload 到新武器
                agent.ammo = wp.mag_size
                agent.max_ammo = wp.mag_size
                agent.reload_delay = wp.reload_frames
                picked_up.append(item)
            else:
                # 背包已滿 → 找到最差武器（damage 最低）
                worst_idx = 0
                worst_dmg = agent.weapon_slots[0].damage if agent.weapon_slots[0] else 0
                for i, slot in enumerate(agent.weapon_slots):
                    d = slot.damage if slot else 0
                    if d < worst_dmg:
                        worst_dmg = d
                        worst_idx = i
                # 地面武器比最差武器更好才拾取
                if wp.damage > worst_dmg:
                    # 丟棄最差武器，產生掉落物放回 ground_items
                    dropped_wp = agent.weapon_slots[worst_idx]
                    if dropped_wp is not None:
                        dropped_item = GroundItem(
                            x=agent.x, y=agent.y,
                            item_type="weapon",
                            weapon_spec=dropped_wp,
                        )
                        ground_items.append(dropped_item)
                    agent.weapon_slots[worst_idx] = wp
                    # 若替換的是目前持有的武器，同步彈藥數據
                    if worst_idx == agent.active_slot:
                        agent.ammo = wp.mag_size
                        agent.max_ammo = wp.mag_size
                        agent.reload_delay = wp.reload_frames
                    picked_up.append(item)

        elif item.item_type == "medkit":
            if agent.medkits < agent.max_medkits:
                agent.medkits += 1
                picked_up.append(item)

        elif item.item_type == "grenade":
            if agent.grenades < agent.max_grenades:
                agent.grenades += 1
                picked_up.append(item)

    # 從原列表中移除已拾取的項目
    for item in picked_up:
        if item in ground_items:
            ground_items.remove(item)

    return picked_up

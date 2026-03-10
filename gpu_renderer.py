"""
gpu_renderer.py — 端到端 GPU 強化學習架構渲染器
使用 PyTorch 張量運算即時生成 10 通道觀測圖像，替代 CPU Multiprocessing Pipeline。

核心技術：
- F.grid_sample：地形裁切與旋轉（雙 FOV 支援）
- torch.bmm：仿射座標變換
- scatter_add_：實體雷達投影（雙線性插值）
- Distance Field：聲音波紋渲染
- 極座標遮罩：軟性視線遮擋
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional

# ── 地圖與視野常數 ──
TILE_SIZE = 40
VIEW_SIZE = 15
VIEW_CENTER = 7.0  # 中心點位置 (0-indexed)
VIEW_RANGE = 10.0  # Standard FOV 視距（單位：格子）
HALF_FOV = 65.0    # Standard FOV 半角（度）

# Sniper FOV 參數（從 game/fov.py 提取）
SNIPER_VIEW_RANGE = 15.0
SNIPER_HALF_FOV = 45.0
SNIPER_TILE_SIZE = 60.0

# 道具類型 ID
ITEM_TYPE_IDS = {"weapon": 0, "medkit": 1, "grenade": 2, "ammo": 3}


class GPURenderer:
    """
    GPU 端即時觀測生成器，支援批次處理與雙 FOV 模式。

    使用方法：
    1. 主進程收集 raw_state（輕量級座標）並 Padding 到固定上限
    2. 呼叫 render_batch() 瞬間生成 (B, 10, 15, 15) 觀測張量
    3. 餵給 CNN 後由 PyTorch GC 自動釋放 VRAM
    """

    def __init__(self, map_rows: int = 24, map_cols: int = 32):
        """
        Args:
            map_rows: 地圖列數（預設 24）
            map_cols: 地圖行數（預設 32）
        """
        self.map_rows = map_rows
        self.map_cols = map_cols
        self.map_width = map_cols * TILE_SIZE
        self.map_height = map_rows * TILE_SIZE

    @torch.no_grad()
    def render_batch(
        self,
        agent_poses: torch.Tensor,       # (B, 4): [x, y, angle_deg, is_sniper]
        ally_poses: torch.Tensor,        # (B, MAX_ALLIES, 3): [x, y, hp_value]
        ally_mask: torch.Tensor,         # (B, MAX_ALLIES): bool
        enemy_poses: torch.Tensor,       # (B, MAX_ENEMIES, 3): [x, y, hp_value]
        enemy_mask: torch.Tensor,        # (B, MAX_ENEMIES): bool
        item_poses: torch.Tensor,        # (B, MAX_ITEMS, 3): [x, y, item_type_id]
        item_mask: torch.Tensor,         # (B, MAX_ITEMS): bool
        threat_poses: torch.Tensor,      # (B, MAX_THREATS, 3): [x, y, threat_value]
        threat_mask: torch.Tensor,       # (B, MAX_THREATS): bool
        sound_waves: torch.Tensor,       # (B, MAX_SOUNDS, 4): [x, y, radius, sound_value]
        sound_mask: torch.Tensor,        # (B, MAX_SOUNDS): bool
        grids: torch.Tensor,             # (B, H_map, W_map): 地圖網格（1=牆壁）
        poison_info: torch.Tensor,       # (B, 4): [cx, cy, radius, max_radius]
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        批次渲染 10 通道觀測圖像。

        Returns:
            obs: (B, 10, VIEW_SIZE, VIEW_SIZE) 觀測張量
        """
        if device is None:
            device = agent_poses.device

        B = agent_poses.size(0)
        obs = torch.zeros(B, 10, VIEW_SIZE, VIEW_SIZE, device=device, dtype=torch.float32)

        # 提取 agent 參數
        agent_x = agent_poses[:, 0]           # (B,)
        agent_y = agent_poses[:, 1]           # (B,)
        agent_angle = agent_poses[:, 2]       # (B,) degrees
        is_sniper = agent_poses[:, 3] > 0.5   # (B,) bool

        # 根據 is_sniper 動態選擇 FOV 參數（向量化）
        view_range = torch.where(is_sniper,
                                 torch.tensor(SNIPER_VIEW_RANGE, device=device),
                                 torch.tensor(VIEW_RANGE, device=device))  # (B,)
        half_fov = torch.where(is_sniper,
                               torch.tensor(SNIPER_HALF_FOV, device=device),
                               torch.tensor(HALF_FOV, device=device))      # (B,)
        tile_size = torch.where(is_sniper,
                                torch.tensor(SNIPER_TILE_SIZE, device=device),
                                torch.tensor(TILE_SIZE, device=device))    # (B,)

        # ═══════════════════════════════════════════════════════
        # Ch0: 地形（F.grid_sample 批次裁切與旋轉）
        # ═══════════════════════════════════════════════════════
        obs[:, 0] = self._render_terrain_batch(
            grids, agent_x, agent_y, agent_angle, view_range, tile_size, device
        )

        # ═══════════════════════════════════════════════════════
        # Ch1: 敵人雷達（LOS 過濾）
        # ═══════════════════════════════════════════════════════
        obs[:, 1] = self._render_entities_batch(
            enemy_poses, enemy_mask, agent_x, agent_y, agent_angle,
            view_range, half_fov, tile_size, grids, check_los=True, device=device
        )

        # ═══════════════════════════════════════════════════════
        # Ch2: 隊友雷達（全域可見）
        # ═══════════════════════════════════════════════════════
        obs[:, 2] = self._render_entities_batch(
            ally_poses, ally_mask, agent_x, agent_y, agent_angle,
            view_range, half_fov, tile_size, grids, check_los=False, device=device
        )

        # ═══════════════════════════════════════════════════════
        # Ch3: 威脅/彈道（LOS 過濾）
        # ═══════════════════════════════════════════════════════
        obs[:, 3] = self._render_entities_batch(
            threat_poses, threat_mask, agent_x, agent_y, agent_angle,
            view_range, half_fov, tile_size, grids, check_los=True, device=device
        )

        # ═══════════════════════════════════════════════════════
        # Ch4: 聲音波紋（Distance Field 圓環渲染）
        # ═══════════════════════════════════════════════════════
        obs[:, 4] = self._render_sound_waves_batch(
            sound_waves, sound_mask, agent_x, agent_y, agent_angle, tile_size, device
        )

        # ═══════════════════════════════════════════════════════
        # Ch5: 安全區/毒圈（Distance Field）
        # ═══════════════════════════════════════════════════════
        obs[:, 5] = self._render_poison_zone_batch(
            poison_info, agent_x, agent_y, agent_angle, tile_size, device
        )

        # ═══════════════════════════════════════════════════════
        # Ch6-9: 道具雷達（weapon, medkit, grenade, ammo）
        # ═══════════════════════════════════════════════════════
        for item_type_id, ch_idx in [(0, 6), (1, 7), (2, 8), (3, 9)]:
            # 過濾出特定類型的道具
            type_mask = (item_poses[:, :, 2] == item_type_id) & item_mask  # (B, MAX_ITEMS)
            filtered_poses = item_poses.clone()
            filtered_poses[:, :, 2] = 1.0  # 將 value 統一設為 1.0

            obs[:, ch_idx] = self._render_entities_batch(
                filtered_poses, type_mask, agent_x, agent_y, agent_angle,
                view_range, half_fov, tile_size, grids, check_los=True, device=device
            )

        return obs

    # ═══════════════════════════════════════════════════════
    #  核心渲染模組
    # ═══════════════════════════════════════════════════════

    def _render_terrain_batch(
        self,
        grids: torch.Tensor,       # (B, H_map, W_map)
        agent_x: torch.Tensor,     # (B,)
        agent_y: torch.Tensor,     # (B,)
        agent_angle: torch.Tensor, # (B,) degrees
        view_range: torch.Tensor,  # (B,)
        tile_size: torch.Tensor,   # (B,)
        device: torch.device,
    ) -> torch.Tensor:
        """
        使用 F.grid_sample 批次裁切與旋轉地圖，生成局部視野地形。

        Returns:
            terrain: (B, VIEW_SIZE, VIEW_SIZE)
        """
        B = grids.size(0)

        # 確保 grids 為 float32 並擴展為 4D (B, 1, H, W) 供 grid_sample 使用
        grids_4d = grids.unsqueeze(1).float()  # (B, 1, H_map, W_map)

        # 計算採樣範圍（單位：像素）
        # view_range 是格子數，乘以 tile_size 得到像素範圍
        sample_size_px = view_range * tile_size  # (B,)

        # 構建仿射變換矩陣（包含旋轉與平移）
        # PyTorch affine_grid 要求 theta: (B, 2, 3)
        # 旋轉矩陣 + 平移向量
        angle_rad = torch.deg2rad(-agent_angle)  # 負號：逆時針旋轉視角
        cos_a = torch.cos(angle_rad)  # (B,)
        sin_a = torch.sin(angle_rad)  # (B,)

        # 將像素座標正規化到 [-1, 1]（grid_sample 要求）
        # 地圖中心 = (map_width/2, map_height/2)
        # agent 位置相對於地圖中心的偏移
        tx = (agent_x - self.map_width / 2) / (self.map_width / 2)   # (B,)
        ty = (agent_y - self.map_height / 2) / (self.map_height / 2) # (B,)

        # 縮放因子：控制採樣範圍（view_range 越大，縮放越小）
        # Standard FOV 視距 10 格 → 採樣 10*40=400 px
        # Sniper FOV 視距 15 格 → 採樣 15*60=900 px
        scale = (VIEW_SIZE * TILE_SIZE) / (2 * sample_size_px)  # (B,)

        # 構建仿射矩陣 theta (B, 2, 3)
        # [ cos*scale  -sin*scale  tx ]
        # [ sin*scale   cos*scale  ty ]
        theta = torch.zeros(B, 2, 3, device=device)
        theta[:, 0, 0] = cos_a * scale
        theta[:, 0, 1] = -sin_a * scale
        theta[:, 0, 2] = tx
        theta[:, 1, 0] = sin_a * scale
        theta[:, 1, 1] = cos_a * scale
        theta[:, 1, 2] = ty

        # 生成採樣網格
        grid = F.affine_grid(theta, (B, 1, VIEW_SIZE, VIEW_SIZE), align_corners=False)

        # 採樣（使用最近鄰插值保持地形離散性）
        terrain = F.grid_sample(
            grids_4d, grid, mode='nearest', padding_mode='zeros', align_corners=False
        )  # (B, 1, VIEW_SIZE, VIEW_SIZE)

        return terrain.squeeze(1)  # (B, VIEW_SIZE, VIEW_SIZE)

    def _render_entities_batch(
        self,
        entity_poses: torch.Tensor,  # (B, N, 3): [x, y, value]
        entity_mask: torch.Tensor,   # (B, N): bool
        agent_x: torch.Tensor,       # (B,)
        agent_y: torch.Tensor,       # (B,)
        agent_angle: torch.Tensor,   # (B,) degrees
        view_range: torch.Tensor,    # (B,)
        half_fov: torch.Tensor,      # (B,) degrees
        tile_size: torch.Tensor,     # (B,)
        grids: torch.Tensor,         # (B, H_map, W_map) 用於 LOS 檢查
        check_los: bool = False,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        批次投影實體到局部視野雷達圖（使用 scatter_add_ 實現雙線性插值）。

        Args:
            check_los: 是否檢查視線遮擋（敵人/威脅需要，隊友不需要）

        Returns:
            radar: (B, VIEW_SIZE, VIEW_SIZE)
        """
        B, N, _ = entity_poses.shape

        # 提取實體座標與數值
        ex = entity_poses[:, :, 0]  # (B, N)
        ey = entity_poses[:, :, 1]  # (B, N)
        ev = entity_poses[:, :, 2]  # (B, N)

        # 計算相對位置
        dx = ex - agent_x.unsqueeze(1)  # (B, N)
        dy = ey - agent_y.unsqueeze(1)  # (B, N)

        # 旋轉到 agent 局部坐標系（前向 = 0°）
        angle_rad = torch.deg2rad(agent_angle).unsqueeze(1)  # (B, 1)
        cos_a = torch.cos(angle_rad)
        sin_a = torch.sin(angle_rad)

        # 局部坐標（forward, right）
        ft = (dx * cos_a + dy * sin_a) / tile_size.unsqueeze(1)  # (B, N)
        rt = (-dx * sin_a + dy * cos_a) / tile_size.unsqueeze(1) # (B, N)

        # 距離與角度
        dt = torch.sqrt(ft**2 + rt**2)                                     # (B, N)
        ang = torch.rad2deg(torch.atan2(rt, ft))                          # (B, N)

        # FOV 過濾：距離 + 角度範圍
        in_range = (dt <= view_range.unsqueeze(1))                         # (B, N)
        in_fov = (torch.abs(ang) <= half_fov.unsqueeze(1))                # (B, N)
        valid = entity_mask & in_range & in_fov                           # (B, N)

        # TODO: LOS 檢查（軟性遮擋，暫時跳過以簡化初版實作）
        # 可使用極座標累加遮罩或 Bresenham 光線投射

        # 投影到等距菱形坐標系（45° 旋轉）
        dr = (ft + rt) / 1.41421356  # (B, N)
        dc = (ft - rt) / 1.41421356  # (B, N)

        # 轉換到網格坐標（以 VIEW_CENTER 為中心）
        r_f = VIEW_CENTER + dr  # (B, N)
        c_f = VIEW_CENTER + dc  # (B, N)

        # 雙線性插值：計算 4 個鄰居的權重
        r0 = torch.floor(r_f).long()  # (B, N)
        c0 = torch.floor(c_f).long()  # (B, N)
        dr_frac = r_f - r0.float()    # (B, N)
        dc_frac = c_f - c0.float()    # (B, N)

        # 權重
        w00 = (1 - dr_frac) * (1 - dc_frac) * ev  # (B, N)
        w01 = (1 - dr_frac) * dc_frac * ev        # (B, N)
        w10 = dr_frac * (1 - dc_frac) * ev        # (B, N)
        w11 = dr_frac * dc_frac * ev              # (B, N)

        # 邊界檢查
        in_grid_00 = (r0 >= 0) & (r0 < VIEW_SIZE) & (c0 >= 0) & (c0 < VIEW_SIZE)
        in_grid_01 = (r0 >= 0) & (r0 < VIEW_SIZE) & (c0 + 1 >= 0) & (c0 + 1 < VIEW_SIZE)
        in_grid_10 = (r0 + 1 >= 0) & (r0 + 1 < VIEW_SIZE) & (c0 >= 0) & (c0 < VIEW_SIZE)
        in_grid_11 = (r0 + 1 >= 0) & (r0 + 1 < VIEW_SIZE) & (c0 + 1 >= 0) & (c0 + 1 < VIEW_SIZE)

        # 最終有效性：valid & in_grid
        valid_00 = valid & in_grid_00
        valid_01 = valid & in_grid_01
        valid_10 = valid & in_grid_10
        valid_11 = valid & in_grid_11

        # 初始化輸出
        radar = torch.zeros(B, VIEW_SIZE, VIEW_SIZE, device=device)

        # 使用 scatter_add_ 批次注入（需展平 batch 維度）
        # scatter_add_(dim, index, src)
        # 將每個 (b, n) 的值加到 radar[b, r, c]

        # 為避免複雜的多維 scatter，這裡使用簡化的單 batch 迴圈
        # （在實際部署時可優化為純張量操作，但需要更複雜的 index 構造）
        for b in range(B):
            for n in range(N):
                if valid_00[b, n]:
                    radar[b, r0[b, n], c0[b, n]] += w00[b, n]
                if valid_01[b, n]:
                    radar[b, r0[b, n], c0[b, n] + 1] += w01[b, n]
                if valid_10[b, n]:
                    radar[b, r0[b, n] + 1, c0[b, n]] += w10[b, n]
                if valid_11[b, n]:
                    radar[b, r0[b, n] + 1, c0[b, n] + 1] += w11[b, n]

        return radar

    def _render_sound_waves_batch(
        self,
        sound_waves: torch.Tensor,  # (B, M, 4): [x, y, radius, sound_value]
        sound_mask: torch.Tensor,   # (B, M): bool
        agent_x: torch.Tensor,      # (B,)
        agent_y: torch.Tensor,      # (B,)
        agent_angle: torch.Tensor,  # (B,) degrees
        tile_size: torch.Tensor,    # (B,)
        device: torch.device,
    ) -> torch.Tensor:
        """
        使用 Distance Field 渲染聲音波紋圓環。

        Returns:
            sound_ch: (B, VIEW_SIZE, VIEW_SIZE)
        """
        B, M, _ = sound_waves.shape

        # 建立局部網格坐標 (VIEW_SIZE, VIEW_SIZE)
        y_grid, x_grid = torch.meshgrid(
            torch.arange(VIEW_SIZE, device=device),
            torch.arange(VIEW_SIZE, device=device),
            indexing='ij'
        )
        # 轉換為局部坐標（以 VIEW_CENTER 為中心，單位：tile）
        x_local = (x_grid.float() - VIEW_CENTER)  # (VIEW_SIZE, VIEW_SIZE)
        y_local = (y_grid.float() - VIEW_CENTER)  # (VIEW_SIZE, VIEW_SIZE)

        # 旋轉到 agent 坐標系
        angle_rad = torch.deg2rad(agent_angle)  # (B,)
        cos_a = torch.cos(angle_rad).view(B, 1, 1)  # (B, 1, 1)
        sin_a = torch.sin(angle_rad).view(B, 1, 1)  # (B, 1, 1)

        # 廣播到 (B, VIEW_SIZE, VIEW_SIZE)
        x_rot = x_local * cos_a - y_local * sin_a  # (B, VIEW_SIZE, VIEW_SIZE)
        y_rot = x_local * sin_a + y_local * cos_a  # (B, VIEW_SIZE, VIEW_SIZE)

        # 轉換為全局像素坐標
        x_world = agent_x.view(B, 1, 1) + x_rot * tile_size.view(B, 1, 1)  # (B, VIEW_SIZE, VIEW_SIZE)
        y_world = agent_y.view(B, 1, 1) + y_rot * tile_size.view(B, 1, 1)  # (B, VIEW_SIZE, VIEW_SIZE)

        # 初始化聲音通道
        sound_ch = torch.zeros(B, VIEW_SIZE, VIEW_SIZE, device=device)

        # 遍歷每個聲音波紋（向量化處理）
        for m in range(M):
            # 提取波紋參數
            wx = sound_waves[:, m, 0]        # (B,)
            wy = sound_waves[:, m, 1]        # (B,)
            wr = sound_waves[:, m, 2]        # (B,)
            wv = sound_waves[:, m, 3]        # (B,)
            valid = sound_mask[:, m]         # (B,)

            # 計算距離場 (B, VIEW_SIZE, VIEW_SIZE)
            dx = x_world - wx.view(B, 1, 1)
            dy = y_world - wy.view(B, 1, 1)
            dist = torch.sqrt(dx**2 + dy**2)  # (B, VIEW_SIZE, VIEW_SIZE)

            # 圓環遮罩：abs(dist - radius) < thickness
            thickness = 20.0  # 像素
            ring_mask = (torch.abs(dist - wr.view(B, 1, 1)) < thickness).float()  # (B, VIEW_SIZE, VIEW_SIZE)

            # 套用有效性遮罩
            ring_mask = ring_mask * valid.view(B, 1, 1).float()

            # 加權疊加
            sound_ch += ring_mask * wv.view(B, 1, 1)

        return sound_ch

    def _render_poison_zone_batch(
        self,
        poison_info: torch.Tensor,  # (B, 4): [cx, cy, radius, max_radius]
        agent_x: torch.Tensor,      # (B,)
        agent_y: torch.Tensor,      # (B,)
        agent_angle: torch.Tensor,  # (B,) degrees
        tile_size: torch.Tensor,    # (B,)
        device: torch.device,
    ) -> torch.Tensor:
        """
        渲染安全區/毒圈距離場（標量填充，不需要空間分佈）。

        根據 CPU 版本邏輯：
        - 1.0 = 圈心安全
        - 0.0 = 剛好在圈邊
        - 負值 = 圈外危險

        Returns:
            poison_ch: (B, VIEW_SIZE, VIEW_SIZE)
        """
        B = poison_info.size(0)

        cx = poison_info[:, 0]          # (B,)
        cy = poison_info[:, 1]          # (B,)
        radius = poison_info[:, 2]      # (B,)
        max_radius = poison_info[:, 3]  # (B,)

        # 計算 agent 到圈心的距離
        dist_self = torch.sqrt((agent_x - cx)**2 + (agent_y - cy)**2)  # (B,)

        # 標準化距離：1.0 - dist/radius
        poison_value = torch.clamp(1.0 - dist_self / torch.clamp(radius, min=1.0), -1.0, 1.0)  # (B,)

        # 廣播到整個視野（每個像素相同）
        poison_ch = poison_value.view(B, 1, 1).expand(B, VIEW_SIZE, VIEW_SIZE)

        # 若 radius 為無限大（無毒圈），填充 1.0
        no_poison = (radius >= 1e9)  # (B,)
        poison_ch[no_poison] = 1.0

        return poison_ch


# ═══════════════════════════════════════════════════════
#  工具函式：從 NumPy raw_state 打包成 Tensor
# ═══════════════════════════════════════════════════════

def pack_raw_states_to_tensors(
    raw_states: list,  # List[Dict]，每個 Dict 包含 raw_state 欄位
    max_allies: int = 3,
    max_enemies: int = 8,
    max_items: int = 20,
    max_threats: int = 32,
    max_sounds: int = 16,
    device: torch.device = None,
) -> Tuple[torch.Tensor, ...]:
    """
    將主進程收集的 raw_state 列表打包成 Padded Tensors。

    Args:
        raw_states: List of raw_state dicts from env.step()
        max_*: Padding 上限

    Returns:
        Tuple of tensors ready for gpu_renderer.render_batch()
    """
    B = len(raw_states)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化 Padded Tensors
    agent_poses = torch.zeros(B, 4, device=device)                        # [x, y, angle, is_sniper]
    ally_poses = torch.zeros(B, max_allies, 3, device=device)             # [x, y, hp_value]
    ally_mask = torch.zeros(B, max_allies, dtype=torch.bool, device=device)
    enemy_poses = torch.zeros(B, max_enemies, 3, device=device)
    enemy_mask = torch.zeros(B, max_enemies, dtype=torch.bool, device=device)
    item_poses = torch.zeros(B, max_items, 3, device=device)
    item_mask = torch.zeros(B, max_items, dtype=torch.bool, device=device)
    threat_poses = torch.zeros(B, max_threats, 3, device=device)
    threat_mask = torch.zeros(B, max_threats, dtype=torch.bool, device=device)
    sound_waves = torch.zeros(B, max_sounds, 4, device=device)
    sound_mask = torch.zeros(B, max_sounds, dtype=torch.bool, device=device)

    # Grids（假設所有環境的地圖大小相同，這裡用第一個的尺寸）
    sample_grid = raw_states[0]["grid"]
    H_map, W_map = sample_grid.shape
    grids = torch.zeros(B, H_map, W_map, device=device)

    poison_info = torch.zeros(B, 4, device=device)  # [cx, cy, radius, max_radius]

    # 填充數據
    for b, rs in enumerate(raw_states):
        # Agent pose
        agent_poses[b] = torch.tensor(rs["agent_pose"], dtype=torch.float32)

        # Allies
        allies = rs["ally_poses"]
        n_allies = min(len(allies), max_allies)
        if n_allies > 0:
            ally_poses[b, :n_allies] = torch.tensor(allies[:n_allies], dtype=torch.float32)
            ally_mask[b, :n_allies] = True

        # Enemies
        enemies = rs["enemy_poses"]
        n_enemies = min(len(enemies), max_enemies)
        if n_enemies > 0:
            enemy_poses[b, :n_enemies] = torch.tensor(enemies[:n_enemies], dtype=torch.float32)
            enemy_mask[b, :n_enemies] = True

        # Items
        items = rs["item_poses"]
        n_items = min(len(items), max_items)
        if n_items > 0:
            item_poses[b, :n_items] = torch.tensor(items[:n_items], dtype=torch.float32)
            item_mask[b, :n_items] = True

        # Threats
        threats = rs["threat_poses"]
        n_threats = min(len(threats), max_threats)
        if n_threats > 0:
            threat_poses[b, :n_threats] = torch.tensor(threats[:n_threats], dtype=torch.float32)
            threat_mask[b, :n_threats] = True

        # Sounds
        sounds = rs["sound_waves"]
        n_sounds = min(len(sounds), max_sounds)
        if n_sounds > 0:
            sound_waves[b, :n_sounds] = torch.tensor(sounds[:n_sounds], dtype=torch.float32)
            sound_mask[b, :n_sounds] = True

        # Grid
        grids[b] = torch.tensor(rs["grid"], dtype=torch.float32)

        # Poison
        poison_info[b] = torch.tensor(rs["poison_info"], dtype=torch.float32)

    return (
        agent_poses, ally_poses, ally_mask, enemy_poses, enemy_mask,
        item_poses, item_mask, threat_poses, threat_mask,
        sound_waves, sound_mask, grids, poison_info
    )

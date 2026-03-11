"""
watch.py — 觀戰腳本（支援多 AI + 通訊向量顯示）
相容重構後的介面：
  - env 回傳 (view, scalars, team_id) 三元組
  - LSTM hidden = (h, c) tuple
  - model.forward → (logits, comm_mu, comm_logstd, feat, new_hidden)
  - 動作維度 = 12（Bernoulli 離散）
  - comm 維度 = 4
"""
import argparse
import math
import os
import time

import numpy as np
import pygame
import torch
from dataclasses import replace
from torch.distributions import Bernoulli, Normal

from ai import ConvLSTM, HIDDEN_SIZE, NUM_COMM, NUM_ACTIONS_DISCRETE
from game import (
    GameEnv, get_stage_spec,
    WIDTH, HEIGHT, FPS, ROWS, COLS, TILE_SIZE,
    VIEW_RANGE, HALF_FOV, FOV_DEGREES,
)
from game.env import NUM_CHANNELS, NUM_SCALARS

FRAME_SKIP = 2

# 各 AI agent 的 FOV 扇形顏色
FOV_COLORS = [
    (80, 220, 255),    # agent 0: 青藍
    (255, 180, 50),    # agent 1: 橙黃
    (100, 255, 150),   # agent 2: 淡綠
    (255, 100, 200),   # agent 3: 粉紅
]


def find_cjk_font() -> str | None:
    """嘗試找出系統上可顯示中文的字型路徑，找不到回傳 None。"""
    import sys, os, glob
    candidates = []

    if sys.platform == "win32":
        windir = os.environ.get("WINDIR", "C:\\Windows")
        for name in ("msjh.ttc", "mingliu.ttc", "simsun.ttc",
                     "msyh.ttc", "kaiu.ttf"):
            candidates.append(os.path.join(windir, "Fonts", name))
    elif sys.platform == "darwin":
        for name in ("PingFang.ttc", "STHeiti Medium.ttc",
                     "Arial Unicode.ttf", "Hiragino Sans GB.ttc"):
            candidates += glob.glob(f"/System/Library/Fonts/**/{name}", recursive=True)
            candidates += glob.glob(f"/Library/Fonts/{name}")
    else:
        try:
            import subprocess
            out = subprocess.check_output(
                ["fc-list", ":lang=zh", "--format=%{file}\n"],
                stderr=subprocess.DEVNULL, text=True)
            candidates += [l.strip() for l in out.splitlines() if l.strip()]
        except Exception:
            pass
        for pattern in (
            "/usr/share/fonts/**/*.ttf",
            "/usr/share/fonts/**/*.ttc",
            "/usr/local/share/fonts/**/*.ttf",
        ):
            candidates += glob.glob(pattern, recursive=True)

    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def make_fonts(sizes: dict) -> dict:
    cjk = find_cjk_font()
    result = {}
    for key, size in sizes.items():
        if cjk:
            try:
                result[key] = pygame.font.Font(cjk, size)
                continue
            except Exception:
                pass
        result[key] = pygame.font.SysFont(None, size)
    return result


SPEED_STEPS = [0.25, 0.5, 1.0, 2.0, 4.0]
DEFAULT_SPEED_IDX = 2

CELL = 8
PAD = 8
LABEL_H = 20
CH_NAMES = ["地形", "敵人", "隊友", "威脅", "聲音", "安全區", "武器", "醫療包", "手榴彈", "彈藥"]


def parse_model_path(ckpt: str) -> str:
    if ckpt == "final":
        return "snn_ep_final.pth"
    if ckpt.endswith(".pth"):
        return ckpt
    return f"snn_ep_{ckpt}.pth"


def load_model(model_path: str, device):
    payload = torch.load(model_path, map_location=device, weights_only=False)
    model = ConvLSTM().to(device)
    if isinstance(payload, dict) and "model_state" in payload:
        model.load_state_dict(payload["model_state"])
        n_ai = payload.get("n_ai", 1)
    else:
        model.load_state_dict(payload)
        n_ai = 1
    model.eval()
    return model, n_ai


def channel_to_surface(data: np.ndarray) -> pygame.Surface:
    ch_px = 15 * CELL
    surf = pygame.Surface((ch_px, ch_px))
    vmax = float(np.abs(data).max()) or 1e-6
    for r in range(15):
        for c in range(15):
            v = float(data[r, c])
            if v > 0:
                t = v / vmax
                color = (int(255 * t), int(180 * t), int(50 * t))
            elif v < 0:
                t = -v / vmax
                color = (int(50 * t), int(100 * t), int(255 * t))
            else:
                color = (20, 20, 30)
            pygame.draw.rect(surf, color, (c * CELL, r * CELL, CELL, CELL))
    return surf


def draw_ai_panel(screen, view_np, x0, y0, font_sm):
    ch_px = 15 * CELL
    col_stride = ch_px + PAD
    row_stride = ch_px + LABEL_H + PAD
    n_ch = min(view_np.shape[0], len(CH_NAMES))
    for ch in range(n_ch):
        col = ch % 2
        row = ch // 2
        x = x0 + col * col_stride
        y = y0 + row * row_stride
        screen.blit(font_sm.render(CH_NAMES[ch], True, (200, 200, 200)), (x, y))
        surf = channel_to_surface(view_np[ch])
        screen.blit(surf, (x, y + LABEL_H))
        pygame.draw.rect(screen, (80, 80, 100), (x, y + LABEL_H, ch_px, ch_px), 1)


def draw_comm_panel(screen, comm_vecs, x0, y0, font_sm):
    """繪製通訊向量棒狀圖"""
    bar_w = 80
    bar_h = 12
    gap = 4

    for ai_idx, cv in enumerate(comm_vecs):
        label = f"A{ai_idx}_comm"
        screen.blit(font_sm.render(label, True, (200, 200, 200)), (x0, y0))
        y0 += 18

        for dim_i in range(len(cv)):
            val = float(cv[dim_i])
            # 背景
            pygame.draw.rect(screen, (40, 40, 50),
                             (x0, y0, bar_w, bar_h))
            # 值
            center_x = x0 + bar_w // 2
            fill_w = int(abs(val) * (bar_w // 2))
            if val >= 0:
                color = (80, 140, 255)  # 藍=正
                pygame.draw.rect(screen, color,
                                 (center_x, y0, fill_w, bar_h))
            else:
                color = (255, 100, 100)  # 紅=負
                pygame.draw.rect(screen, color,
                                 (center_x - fill_w, y0, fill_w, bar_h))
            # 中心線
            pygame.draw.line(screen, (150, 150, 150),
                             (center_x, y0), (center_x, y0 + bar_h), 1)
            y0 += bar_h + gap

        y0 += 8


def _unpack_state(state_tuple):
    """
    解包 env 回傳的 state，相容新舊格式：
      新格式 (view, scalars, team_id) → return view, scalars, team_id
      舊格式 (view, scalars)          → return view, scalars, 0
    """
    if len(state_tuple) == 3:
        return state_tuple[0], state_tuple[1], int(state_tuple[2])
    return state_tuple[0], state_tuple[1], 0


def watch_ai(ckpt: str, stage_id: int,
             max_frames: int = 1200,
             show_ai_view: bool = False,
             show_comm: bool = False):

    model_path = parse_model_path(ckpt)
    if not os.path.exists(model_path):
        print(f"找不到模型: {model_path}")
        return

    device = torch.device("cpu")
    model, n_ai = load_model(model_path, device)
    stage_spec = get_stage_spec(stage_id)

    print(f"載入模型  : {model_path} (n_ai={n_ai})")
    print(f"觀戰階段  : Stage {stage_id} - {stage_spec.name}")
    print("操作說明  : [Space] 暫停  []] 加速  [[] 減速  [Esc] 離開")

    ch_px = 15 * CELL
    panel_w = 2 * (ch_px + PAD) + PAD if show_ai_view else 0
    if show_comm:
        panel_w = max(panel_w, 120)
    total_w = WIDTH + panel_w

    pygame.init()
    screen = pygame.display.set_mode((total_w, HEIGHT))
    pygame.display.set_caption(f"Watch – Stage {stage_id}  {stage_spec.name}")
    clock = pygame.time.Clock()
    fonts = make_fonts({"normal": 24, "small": 20, "big": 36})
    font = fonts["normal"]
    font_sm = fonts["small"]
    font_big = fonts["big"]

    env = GameEnv(render_mode=False, stage_id=stage_id, show_fov=True,
                  n_learning_agents=n_ai)
    env.screen = screen
    env.font = font
    env.stage_spec = replace(env.stage_spec, max_frames=max_frames)

    # ── Reset 並解包 state ──
    result = env.reset()
    if n_ai == 1:
        states = [result]          # result = (view, scalars, team_id)
    else:
        states = result            # result = [(view, scalars, tid), ...]

    # ── LSTM hidden per agent: (h, c) 各 (1, 1, HIDDEN_SIZE) ──
    hiddens = [
        (torch.zeros(1, 1, HIDDEN_SIZE, device=device),
         torch.zeros(1, 1, HIDDEN_SIZE, device=device))
        for _ in range(n_ai)
    ]

    # 通訊向量（上一步輸出，供下一步做 comm_in）
    comm_vecs = [np.zeros(NUM_COMM, dtype=np.float32) for _ in range(n_ai)]

    # 相機系統（簡單平移，不旋轉）
    camera_x = WIDTH / 2   # 相機中心 X（世界坐標）
    camera_y = HEIGHT / 2  # 相機中心 Y（世界坐標）
    camera_mode = "follow"  # "follow" 或 "free"
    follow_agent_index = 0  # 跟隨哪個 learning agent
    camera_dragging = False
    drag_start_pos = (0, 0)
    drag_start_camera = (0.0, 0.0)

    done = False
    step = 0
    actions = [[0.0] * NUM_ACTIONS_DISCRETE for _ in range(n_ai)]
    paused = False
    speed_idx = DEFAULT_SPEED_IDX
    v0, s0, _ = _unpack_state(states[0])
    last_view_np = v0.copy()
    last_reward = 0.0
    last_info: dict = {}

    def speed_label() -> str:
        s = SPEED_STEPS[speed_idx]
        return f"{s:.2g}x"

    def world_to_screen(wx, wy):
        """世界坐標 → 屏幕坐標（簡單平移，不旋轉）"""
        sx = wx - camera_x + WIDTH / 2
        sy = wy - camera_y + HEIGHT / 2
        return (sx, sy)

    def update_camera():
        """更新相機位置"""
        nonlocal camera_x, camera_y
        if camera_mode == "follow" and env.learning_agents:
            idx = follow_agent_index % len(env.learning_agents)
            agent = env.learning_agents[idx]
            if not agent.truly_dead():
                camera_x = agent.x
                camera_y = agent.y

    def draw_frame():
        # 更新相機位置
        update_camera()

        pygame.draw.rect(screen, (20, 20, 30), (0, 0, WIDTH, HEIGHT))

        # 繪製地形（使用相機偏移）
        for r in range(env.grid_rows):
            for c in range(env.grid_cols):
                if env.grid[r, c] == 1:
                    wx, wy = c * TILE_SIZE, r * TILE_SIZE
                    sx, sy = world_to_screen(wx, wy)
                    pygame.draw.rect(screen, (100, 100, 120),
                                     (int(sx), int(sy), TILE_SIZE, TILE_SIZE))

        # 繪製毒圈（如果有）
        if hasattr(env.stage_spec, 'has_poison_zone') and env.stage_spec.has_poison_zone:
            if env.poison_radius < float('inf'):
                center_screen = world_to_screen(env.poison_cx, env.poison_cy)
                pygame.draw.circle(screen, (255, 60, 60),
                                 (int(center_screen[0]), int(center_screen[1])),
                                 int(env.poison_radius), 2)

        # 地面道具（使用相機偏移）
        for gi in env.ground_items:
            sx, sy = world_to_screen(gi.x, gi.y)
            if gi.item_type == "weapon":
                color = (180, 180, 60)
            elif gi.item_type == "medkit":
                color = (60, 220, 60)
            else:
                color = (220, 60, 60)
            pygame.draw.circle(screen, color, (int(sx), int(sy)), 4)

        # 手榴彈（使用相機偏移）
        for g in env.grenades_list:
            if not g.exploded:
                sx, sy = world_to_screen(g.x, g.y)
                pygame.draw.circle(screen, (255, 140, 0), (int(sx), int(sy)), 4)

        # 子彈（使用相機偏移）
        for p in env.projectiles:
            sx, sy = world_to_screen(p.x, p.y)
            pygame.draw.circle(screen, (255, 220, 50), (int(sx), int(sy)), p.radius)

        # Agents（使用相機偏移）
        for a in env.all_agents:
            if a.alive():
                sx, sy = world_to_screen(a.x, a.y)
                # 繪製 agent 身體
                pygame.draw.circle(screen, a.color, (int(sx), int(sy)), a.radius)
                # 繪製方向指示
                rad = math.radians(a.angle)
                ex_world = a.x + math.cos(rad) * a.radius * 1.5
                ey_world = a.y + math.sin(rad) * a.radius * 1.5
                ex_screen, ey_screen = world_to_screen(ex_world, ey_world)
                pygame.draw.line(screen, (255, 255, 0),
                               (int(sx), int(sy)),
                               (int(ex_screen), int(ey_screen)), 2)

                # 跟隨目標高亮
                if (camera_mode == "follow" and
                    a in env.learning_agents and
                    env.learning_agents.index(a) == follow_agent_index):
                    pygame.draw.circle(screen, (255, 255, 100),
                                     (int(sx), int(sy)), a.radius + 5, 2)

        # 多 AI 各用不同 FOV 顏色（使用相機偏移）
        for i, la in enumerate(env.learning_agents):
            if la.alive():
                color = FOV_COLORS[i % len(FOV_COLORS)]
                _draw_fov_with_camera(screen, la, color, env.show_fov, world_to_screen)

        ai = env.ai_agent
        wp_name = ""
        if ai.active_weapon:
            wp_name = ai.active_weapon.name
        hud1 = (f"Stage {stage_id}  {stage_spec.name}  |"
                f"  HP:{ai.hp}  Ammo:{ai.ammo}  Weapon:{wp_name}"
                f"  AmmoBox:{ai.ammo_boxes}")
        hud2 = (f"EnemyAlive:{len(env._alive_enemies())}"
                f"  AllyAlive:{len([x for x in env.team_agents if x.alive()])}")
        screen.blit(font.render(hud1, True, (255, 255, 255)), (10, 8))
        screen.blit(font.render(hud2, True, (255, 255, 255)), (10, 30))

        ckpt_short = os.path.basename(ckpt) if len(ckpt) > 20 else ckpt
        hud3 = (f"Ckpt:{ckpt_short}  "
                f"Speed:{speed_label()}  "
                f"Frame:{step}/{max_frames}  "
                f"R:{last_reward:.2f}  "
                f"Win:{int(last_info.get('ai_win', False))}")
        screen.blit(font.render(hud3, True, (255, 220, 50)), (10, HEIGHT - 50))

        # 相機模式指示
        if camera_mode == "follow":
            cam_text = f"Camera: FOLLOW Agent {follow_agent_index + 1}"
            cam_color = (100, 255, 100)
        else:
            cam_text = f"Camera: FREE ({int(camera_x)}, {int(camera_y)})"
            cam_color = (255, 200, 100)
        screen.blit(font_sm.render(cam_text, True, cam_color), (10, 52))

        ctrl = font_sm.render(
            "[Space] 暫停/繼續    []] 加速    [[] 減速    [Tab] 相機模式    [1-9] 切換AI    [Esc] 離開",
            True, (130, 130, 130))
        screen.blit(ctrl, (10, HEIGHT - 26))

        if paused:
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 120))
            screen.blit(overlay, (0, 0))
            msg = font_big.render(
                f"PAUSED ({speed_label()})  [Space] to resume",
                True, (255, 220, 50))
            screen.blit(msg, (WIDTH // 2 - msg.get_width() // 2,
                               HEIGHT // 2 - msg.get_height() // 2))

        if panel_w > 0:
            pygame.draw.rect(screen, (25, 25, 35), (WIDTH, 0, panel_w, HEIGHT))

        if show_ai_view:
            screen.blit(font.render("AI 觀測層", True, (200, 200, 255)),
                        (WIDTH + PAD, PAD))
            draw_ai_panel(screen, last_view_np, WIDTH + PAD, PAD + 26, font_sm)

        if show_comm:
            comm_y = PAD + 26
            if show_ai_view:
                comm_y += 3 * (15 * CELL + LABEL_H + PAD) + 10
            draw_comm_panel(screen, comm_vecs, WIDTH + PAD, comm_y, font_sm)

        pygame.display.flip()

    accumulated = 0.0
    last_tick = time.perf_counter()

    while True:
        speed = SPEED_STEPS[speed_idx]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    last_tick = time.perf_counter()
                    accumulated = 0.0
                elif event.key == pygame.K_RIGHTBRACKET:
                    speed_idx = min(speed_idx + 1, len(SPEED_STEPS) - 1)
                elif event.key == pygame.K_LEFTBRACKET:
                    speed_idx = max(speed_idx - 1, 0)
                # 相機控制
                elif event.key == pygame.K_TAB:
                    camera_mode = "free" if camera_mode == "follow" else "follow"
                elif pygame.K_1 <= event.key <= pygame.K_9:
                    agent_idx = event.key - pygame.K_1
                    if agent_idx < len(env.learning_agents):
                        follow_agent_index = agent_idx
                        camera_mode = "follow"
                # 方向鍵移動相機（自由模式）
                elif camera_mode == "free":
                    move_speed = 40
                    if event.key == pygame.K_UP:
                        camera_y -= move_speed
                    elif event.key == pygame.K_DOWN:
                        camera_y += move_speed
                    elif event.key == pygame.K_LEFT:
                        camera_x -= move_speed
                    elif event.key == pygame.K_RIGHT:
                        camera_x += move_speed
            # 鼠標拖動
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # 左鍵
                    camera_dragging = True
                    drag_start_pos = event.pos
                    drag_start_camera = (camera_x, camera_y)
                    camera_mode = "free"
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    camera_dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if camera_dragging:
                    dx = event.pos[0] - drag_start_pos[0]
                    dy = event.pos[1] - drag_start_pos[1]
                    camera_x = drag_start_camera[0] - dx
                    camera_y = drag_start_camera[1] - dy

        now = time.perf_counter()
        dt = now - last_tick
        last_tick = now

        if not paused and not done:
            accumulated += dt * speed
            game_frame_time = 1.0 / FPS

            while accumulated >= game_frame_time and not done:
                accumulated -= game_frame_time

                if step % FRAME_SKIP == 0:
                    new_actions = []
                    for i in range(n_ai):
                        view_i, sc_i, tid_i = _unpack_state(states[i])

                        with torch.no_grad():
                            # 視覺 tensor: (1, 6, 15, 15)
                            s0_t = torch.tensor(
                                view_i, dtype=torch.float32
                            ).unsqueeze(0).to(device)
                            # 純量 tensor: (1, 22)
                            s1_t = torch.tensor(
                                sc_i, dtype=torch.float32
                            ).unsqueeze(0).to(device)

                            # comm_in: 收集同隊其他 agent 上一步的 comm 輸出
                            mate_comms = []
                            for j in range(n_ai):
                                if j == i:
                                    continue
                                _, _, tid_j = _unpack_state(states[j])
                                if tid_j == tid_i:
                                    mate_comms.append(comm_vecs[j])
                            if mate_comms:
                                comm_in_t = torch.tensor(
                                    np.array(mate_comms), dtype=torch.float32
                                ).unsqueeze(0).to(device)  # (1, K, 4)
                            else:
                                comm_in_t = None

                            # 前向傳播
                            logits, mu, logstd, _, new_hidden = model(
                                s0_t, s1_t, hiddens[i], comm_in_t)

                            # 套用 Action Masking（與 train.py 一致）
                            mask = env.learning_agents[i].get_action_mask()
                            mask_t = torch.tensor(mask, dtype=torch.bool, device=device)
                            logits_masked = logits[0].masked_fill(~mask_t, -1e9)
                            # 離散動作: Bernoulli 取樣
                            probs = torch.sigmoid(logits_masked)
                            act = (probs > 0.5).float().cpu().tolist()

                            # 更新通訊向量
                            comm_vecs[i] = mu[0].cpu().numpy()

                            # 更新 hidden state
                            hiddens[i] = new_hidden

                        new_actions.append(act)

                    actions = new_actions
                    v0, _, _ = _unpack_state(states[0])
                    last_view_np = v0.copy()

                # env.step — 回傳格式依 n_ai 自動切換
                result = env.step(actions, frame_skip=1)
                if n_ai == 1:
                    state_out, last_reward, done, last_info = result
                    states = [state_out]   # state_out = (view, scalars, team_id)
                else:
                    states_out, rewards_out, done, last_info = result
                    states = states_out    # list of (view, scalars, team_id)
                    last_reward = (sum(rewards_out) / len(rewards_out)
                                   if rewards_out else 0.0)
                step += 1

        draw_frame()
        clock.tick(FPS)

        if done:
            result_str = ("勝利" if last_info.get("ai_win")
                          else ("失敗" if last_info.get("ai_lost")
                                else "平局"))
            draw_frame()
            end_msg = font_big.render(
                f"結束：{result_str}    [任意鍵] 離開", True, (255, 220, 50))
            screen.blit(end_msg, (WIDTH // 2 - end_msg.get_width() // 2,
                                   HEIGHT // 2 - end_msg.get_height() // 2))
            pygame.display.flip()

            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type in (pygame.QUIT, pygame.KEYDOWN,
                                      pygame.MOUSEBUTTONDOWN):
                        waiting = False
                clock.tick(10)

            pygame.quit()
            print(f"觀戰結束：{result_str}")
            return


def _draw_fov_with_camera(screen, agent, color, show_fov, world_to_screen):
    """繪製指定顏色的 FOV 扇形（使用相機偏移）"""
    if not show_fov:
        return
    rad = math.radians(agent.angle)
    fwd_x, fwd_y = math.cos(rad), math.sin(rad)
    rgt_x, rgt_y = math.cos(rad + math.pi / 2), math.sin(rad + math.pi / 2)
    view_r = float(VIEW_RANGE)
    half_fov_val = HALF_FOV
    fov_degrees_val = FOV_DEGREES
    tile_size_val = float(TILE_SIZE)
    if agent.active_weapon and getattr(agent.active_weapon, 'name', '') == 'sniper':
        from game.fov import SNIPER_VIEW_RANGE, SNIPER_HALF_FOV, SNIPER_FOV_DEGREES, SNIPER_TILE_SIZE
        view_r = float(SNIPER_VIEW_RANGE)
        half_fov_val = float(SNIPER_HALF_FOV)
        fov_degrees_val = float(SNIPER_FOV_DEGREES)
        tile_size_val = float(SNIPER_TILE_SIZE)

    # 轉換所有點到屏幕坐標
    center_screen = world_to_screen(agent.x, agent.y)
    pts = [center_screen]
    steps = 60
    for i in range(steps + 1):
        deg_rel = -half_fov_val + (fov_degrees_val * i / steps)
        rad_rel = math.radians(deg_rel)
        cos_rel = math.cos(rad_rel)
        sin_rel = math.sin(rad_rel)
        ft_val = view_r * cos_rel
        rt_val = view_r * sin_rel
        wx = agent.x + (fwd_x * ft_val + rgt_x * rt_val) * tile_size_val
        wy = agent.y + (fwd_y * ft_val + rgt_y * rt_val) * tile_size_val
        screen_pos = world_to_screen(wx, wy)
        pts.append(screen_pos)
    pts.append(center_screen)
    if len(pts) > 2:
        pygame.draw.lines(screen, color, True, pts, 1)


def main():
    parser = argparse.ArgumentParser(description="觀戰腳本")
    parser.add_argument("--ckpt", default="final",
                        help="checkpoint 名稱：final / 數字 / 完整路徑")
    parser.add_argument("--stage", type=int, default=0,
                        choices=list(range(7)),
                        help="觀看的訓練階段 (0–6)")
    parser.add_argument("--max_frames", type=int, default=1200,
                        help="遊戲最大幀數上限")
    parser.add_argument("--ai_view", action="store_true",
                        help="在右側顯示 AI 觀測層")
    parser.add_argument("--show_comm", action="store_true",
                        help="在右側顯示通訊向量棒狀圖")
    args = parser.parse_args()

    watch_ai(
        ckpt=args.ckpt,
        stage_id=args.stage,
        max_frames=args.max_frames,
        show_ai_view=args.ai_view,
        show_comm=args.show_comm,
    )


if __name__ == "__main__":
    main()

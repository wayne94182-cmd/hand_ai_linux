import argparse
import os
import time

import numpy as np
import pygame
import torch
from dataclasses import replace

from ai import ConvSNN, HIDDEN_SIZE
from game import (
    GameEnv, get_stage_spec,
    WIDTH, HEIGHT, FPS, ROWS, COLS, TILE_SIZE,
)

FRAME_SKIP = 2


def find_cjk_font() -> str | None:
    """嘗試找出系統上可顯示中文的字型路徑，找不到回傳 None。"""
    import sys, os, glob
    candidates = []

    if sys.platform == "win32":
        # Windows 常見中文字型
        windir = os.environ.get("WINDIR", "C:\\Windows")
        for name in ("msjh.ttc", "mingliu.ttc", "simsun.ttc",
                     "msyh.ttc", "kaiu.ttf"):
            candidates.append(os.path.join(windir, "Fonts", name))

    elif sys.platform == "darwin":
        # macOS
        for name in ("PingFang.ttc", "STHeiti Medium.ttc",
                     "Arial Unicode.ttf", "Hiragino Sans GB.ttc"):
            candidates += glob.glob(f"/System/Library/Fonts/**/{name}", recursive=True)
            candidates += glob.glob(f"/Library/Fonts/{name}")

    else:
        # Linux：先查 fc-list，再 fallback 到常見路徑
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
    """
    回傳 {key: pygame.font.Font} 的字典。
    sizes = {"normal": 24, "small": 20, "big": 36}
    """
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

# 可切換的倍速列表，按 [] 鍵循環
SPEED_STEPS = [0.25, 0.5, 1.0, 2.0, 4.0]
DEFAULT_SPEED_IDX = 2  # 預設 1.0x

# AI 觀測層面板
CELL     = 8
PAD      = 8
LABEL_H  = 20
CH_NAMES = ["地形 (Terrain)", "敵人 (Enemy)", "子彈 (Bullet)", "隊友 (Ally)"]


def parse_model_path(ckpt: str) -> str:
    if ckpt == "final":
        return "snn_ep_final.pth"
    if ckpt.endswith(".pth"):
        return ckpt
    return f"snn_ep_{ckpt}.pth"


def load_model(model_path: str, device):
    payload = torch.load(model_path, map_location=device)
    model = ConvSNN().to(device)
    if isinstance(payload, dict) and "model_state" in payload:
        model.load_state_dict(payload["model_state"])
    else:
        model.load_state_dict(payload)
    model.eval()
    return model


def channel_to_surface(data: np.ndarray) -> pygame.Surface:
    ch_px = 15 * CELL
    surf  = pygame.Surface((ch_px, ch_px))
    vmax  = float(np.abs(data).max()) or 1e-6
    for r in range(15):
        for c in range(15):
            v = float(data[r, c])
            if v > 0:
                t     = v / vmax
                color = (int(255 * t), int(180 * t), int(50 * t))
            elif v < 0:
                t     = -v / vmax
                color = (int(50 * t), int(100 * t), int(255 * t))
            else:
                color = (20, 20, 30)
            pygame.draw.rect(surf, color, (c * CELL, r * CELL, CELL, CELL))
    return surf


def draw_ai_panel(screen, view_np, x0, y0, font_sm):
    ch_px      = 15 * CELL
    col_stride = ch_px + PAD
    row_stride = ch_px + LABEL_H + PAD
    for ch in range(4):
        col = ch % 2
        row = ch // 2
        x   = x0 + col * col_stride
        y   = y0 + row * row_stride
        screen.blit(font_sm.render(CH_NAMES[ch], True, (200, 200, 200)), (x, y))
        surf = channel_to_surface(view_np[ch])
        screen.blit(surf, (x, y + LABEL_H))
        pygame.draw.rect(screen, (80, 80, 100), (x, y + LABEL_H, ch_px, ch_px), 1)


def watch_ai(ckpt: str, stage_id: int,
             max_frames: int = 1200,
             show_ai_view: bool = False):

    model_path = parse_model_path(ckpt)
    if not os.path.exists(model_path):
        print(f"找不到模型: {model_path}")
        return

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")   # 不跟 train 搶 GPU
    model  = load_model(model_path, device)
    stage_spec = get_stage_spec(stage_id)

    print(f"載入模型  : {model_path}")
    print(f"觀戰階段  : Stage {stage_id} - {stage_spec.name}")
    print("操作說明  : [Space] 暫停  [] 加速  [] 減速  [Esc] 離開")

    ch_px   = 15 * CELL
    panel_w = 2 * (ch_px + PAD) + PAD if show_ai_view else 0
    total_w = WIDTH + panel_w

    pygame.init()
    screen = pygame.display.set_mode((total_w, HEIGHT))
    pygame.display.set_caption(f"Watch – Stage {stage_id}  {stage_spec.name}")
    clock    = pygame.time.Clock()
    fonts    = make_fonts({"normal": 24, "small": 20, "big": 36})
    font     = fonts["normal"]
    font_sm  = fonts["small"]
    font_big = fonts["big"]

    env = GameEnv(render_mode=False, stage_id=stage_id, show_fov=True)
    env.screen     = screen
    env.font       = font
    env.stage_spec = replace(env.stage_spec, max_frames=max_frames)

    state = env.reset()

    h_ai    = torch.zeros(1, 1, HIDDEN_SIZE, device=device)
    h_enemy = torch.zeros(1, 1, HIDDEN_SIZE, device=device)

    done         = False
    step         = 0
    action       = [0.0] * 9
    enemy_action = None
    paused       = False
    speed_idx    = DEFAULT_SPEED_IDX
    last_view_np = state[0].copy()
    last_reward  = 0.0
    last_info: dict = {}

    def speed_label() -> str:
        s = SPEED_STEPS[speed_idx]
        return f"{s:.2g}x"

    def draw_frame():
        pygame.draw.rect(screen, (20, 20, 30), (0, 0, WIDTH, HEIGHT))

        for r in range(ROWS):
            for c in range(COLS):
                if env.grid[r, c] == 1:
                    pygame.draw.rect(screen, (100, 100, 120),
                                     (c * TILE_SIZE, r * TILE_SIZE,
                                      TILE_SIZE, TILE_SIZE))
        for p in env.projectiles:
            p.draw(screen)
        for a in env.all_agents:
            if a.alive():
                env._draw_agent(a)
        env._draw_fov(env.ai_agent)

        ai   = env.ai_agent
        hud1 = (f"Stage {stage_id}  {stage_spec.name}  |"
                f"  HP:{ai.hp}  Ammo:{ai.ammo}  DashCD:{ai.dash_cd}")
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

        ctrl = font_sm.render(
            "[Space] 暫停/繼續    []] 加速    [[] 減速    [Esc] 離開",
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

        if show_ai_view:
            pygame.draw.rect(screen, (25, 25, 35), (WIDTH, 0, panel_w, HEIGHT))
            screen.blit(font.render("AI 觀測層", True, (200, 200, 255)),
                        (WIDTH + PAD, PAD))
            draw_ai_panel(screen, last_view_np, WIDTH + PAD, PAD + 26, font_sm)

        pygame.display.flip()

    # 用累積時間來決定本幀要推進幾步遊戲邏輯，
    # 這樣低倍速時不會丟幀，高倍速時也不會跑太快。
    accumulated = 0.0
    last_tick   = time.perf_counter()

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
                    last_tick = time.perf_counter()   # 暫停後重置，避免解除後爆衝
                    accumulated = 0.0
                elif event.key == pygame.K_RIGHTBRACKET:  # ] 加速
                    speed_idx = min(speed_idx + 1, len(SPEED_STEPS) - 1)
                elif event.key == pygame.K_LEFTBRACKET:   # [ 減速
                    speed_idx = max(speed_idx - 1, 0)

        now     = time.perf_counter()
        dt      = now - last_tick
        last_tick = now

        if not paused and not done:
            # 把渲染幀的實際時間 × 倍速 累積成「遊戲時間」
            accumulated += dt * speed

            # 每累積夠 1/FPS 秒的遊戲時間，就推進一步遊戲邏輯
            game_frame_time = 1.0 / FPS
            while accumulated >= game_frame_time and not done:
                accumulated -= game_frame_time

                if step % FRAME_SKIP == 0:
                    with torch.no_grad():
                        s0 = torch.tensor(
                            state[0], dtype=torch.float32).unsqueeze(0).to(device)
                        s1 = torch.tensor(
                            state[1], dtype=torch.float32).unsqueeze(0).to(device)
                        logits, _, h_ai = model(s0, s1, h_ai)
                        probs  = torch.sigmoid(logits[0])
                        action = (probs > 0.5).float().cpu().tolist()
                        last_view_np = state[0].copy()

                        if stage_id == 5:
                            logits_e, _, h_enemy = model(s0, s1, h_enemy)
                            probs_e      = torch.sigmoid(logits_e[0])
                            enemy_action = (probs_e > 0.5).float().cpu().tolist()

                state, last_reward, done, last_info = env.step(
                    action, enemy_ai_action=enemy_action, frame_skip=1)
                step += 1

        draw_frame()
        # 渲染幀率固定在 FPS，倍速由遊戲邏輯步數控制
        clock.tick(FPS)

        if done:
            result = ("勝利" if last_info.get("ai_win")
                      else ("失敗" if last_info.get("ai_lost")
                            else "平局"))
            draw_frame()
            end_msg = font_big.render(
                f"結束：{result}    [任意鍵] 離開", True, (255, 220, 50))
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
            print(f"觀戰結束：{result}")
            return


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
                        help="在右側顯示 AI 四通道觀測層")
    args = parser.parse_args()

    watch_ai(
        ckpt         = args.ckpt,
        stage_id     = args.stage,
        max_frames   = args.max_frames,
        show_ai_view = args.ai_view,
    )


if __name__ == "__main__":
    main()

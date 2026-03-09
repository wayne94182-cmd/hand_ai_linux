import argparse
import math
import os
import pygame
from dataclasses import replace

from game import (
    GameEnv, get_stage_spec,
    WIDTH, HEIGHT, FPS, ROWS, COLS, TILE_SIZE,
    VIEW_RANGE, HALF_FOV, FOV_DEGREES,
)
from watch import make_fonts, FOV_COLORS, _draw_fov_color

def play_game(stage_id: int, n_ai: int = 1, max_frames: int = 5400, weapon_name: str = None):
    stage_spec = get_stage_spec(stage_id)
    print(f"遊玩階段  : Stage {stage_id} - {stage_spec.name}")
    if weapon_name:
        print(f"初始武器  : {weapon_name}")
    print("操作說明:")
    print("  W/A/S/D      : 上下左右移動")
    print("  Q/E 或 左右鍵: 視野旋轉 (逆時針/順時針)")
    print("  Ctrl         : 專注模式 (降低旋轉速度)")
    print("  Shift        : 衝刺 (Dash)")
    print("  滑鼠左鍵     : 開火 (Attack)")
    print("  Tab / 2      : 切換武器 (Switch Weapon)")
    print("  H / 3        : 使用醫療包 (Use Medkit)")
    print("  G / 4        : 投擲手榴彈 (Throw Grenade)")
    print("  Z            : 丟棄武器 (Drop Weapon)")
    print("  X            : 丟棄醫療包 (Drop Medkit)")
    print("  C            : 丟棄手榴彈 (Drop Grenade)")
    print("  V            : 丟棄彈藥箱 (Drop Ammo)")
    print("  [Esc]        : 離開")

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"Play – Stage {stage_id}  {stage_spec.name}")
    clock = pygame.time.Clock()
    fonts = make_fonts({"normal": 24, "small": 20, "big": 36})
    font = fonts["normal"]
    font_sm = fonts["small"]
    font_big = fonts["big"]

    # 建立環境
    # n_learning_agents 會根據 n_ai 設定。玩家固定操作 learning_agents[0]
    env = GameEnv(render_mode=False, stage_id=stage_id, show_fov=True,
                  n_learning_agents=n_ai)
    env.screen = screen
    env.font = font
    env.stage_spec = replace(env.stage_spec, max_frames=max_frames)

    env.reset()

    # 設置初始武器
    if weapon_name:
        from game.items import PISTOL, RIFLE, SHOTGUN, SNIPER
        w_map = {"pistol": PISTOL, "rifle": RIFLE, "shotgun": SHOTGUN, "sniper": SNIPER}
        wp = w_map.get(weapon_name.lower())
        if wp:
            player = env.learning_agents[0]
            player.weapon_slots = [wp]
            player.active_slot = 0
            player.ammo = wp.mag_size
            player.max_ammo = wp.mag_size
            player.reload_delay = wp.reload_frames
            player.reload_progress = 0

    done = False
    step = 0
    paused = False
    last_reward = 0.0
    last_info = {}

    def draw_frame():
        pygame.draw.rect(screen, (20, 20, 30), (0, 0, WIDTH, HEIGHT))

        # 畫地圖障礙物
        for r in range(ROWS):
            for c in range(COLS):
                if env.grid[r, c] == 1:
                    pygame.draw.rect(screen, (100, 100, 120),
                                     (c * TILE_SIZE, r * TILE_SIZE,
                                      TILE_SIZE, TILE_SIZE))

        # 地面道具
        for gi in env.ground_items:
            if gi.item_type == "weapon":
                color = (180, 180, 60)
            elif gi.item_type == "medkit":
                color = (60, 220, 60)
            else:
                color = (220, 60, 60)
            pygame.draw.circle(screen, color, (int(gi.x), int(gi.y)), 4)

        # 手榴彈
        for g in env.grenades_list:
            if not g.exploded:
                pygame.draw.circle(screen, (255, 140, 0),
                                   (int(g.x), int(g.y)), 4)

        for p in env.projectiles:
            p.draw(screen)
        for a in env.all_agents:
            if a.alive():
                env._draw_agent(a)

        # FOV 扇形
        for i, la in enumerate(env.learning_agents):
            if la.alive():
                color = FOV_COLORS[i % len(FOV_COLORS)]
                _draw_fov_color(screen, la, color, env.show_fov)

        ai = env.learning_agents[0]
        wp_name = ""
        if ai.active_weapon:
            wp_name = ai.active_weapon.name
        
        hud1 = (f"Stage {stage_id} {stage_spec.name} | "
                f"HP:{int(ai.hp)}/{int(ai.max_hp)}  Ammo:{ai.ammo}  Wt:{wp_name}"
                f"  AmmoBox:{ai.ammo_boxes}")
        hud2 = (f"Medkit:{ai.medkits}  Grenade:{ai.grenades} | "
                f"Enemy:{len(env._alive_enemies())}  Ally:{len([x for x in env.team_agents if x.alive()])}")
        
        if ai.is_downed():
            hud1 += f" [DOWNED! Revive progress: {ai.revive_progress}/{ai.revive_frames}]"
        elif ai.heal_progress > 0:
            hud1 += f" [Healing {ai.heal_progress}/{ai.heal_frames}]"
        elif ai.reload_progress > 0:
            hud1 += f" [Reloading]"
            
        screen.blit(font.render(hud1, True, (255, 255, 255)), (10, 8))
        screen.blit(font.render(hud2, True, (255, 255, 255)), (10, 30))

        hud3 = (f"Frame:{step}/{max_frames}  "
                f"R:{last_reward:.2f}  "
                f"Win:{int(last_info.get('ai_win', False))}")
        screen.blit(font.render(hud3, True, (255, 220, 50)), (10, HEIGHT - 50))

        if paused:
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 120))
            screen.blit(overlay, (0, 0))
            msg = font_big.render("PAUSED  [Space] to resume", True, (255, 220, 50))
            screen.blit(msg, (WIDTH // 2 - msg.get_width() // 2, HEIGHT // 2 - msg.get_height() // 2))

        pygame.display.flip()

    # 追蹤按住狀態，這幾個功能通常只要觸發一次
    action_throw_grenade = 0.0
    action_switch_weapon = 0.0
    action_use_medkit = 0.0
    action_drop_weapon = 0.0
    action_drop_medkit = 0.0
    action_drop_grenade = 0.0
    action_drop_ammo = 0.0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return
                elif event.key == pygame.K_SPACE:
                    pass  # 保留 Space 給另外處理，或者可以在這暫停，但空白現在是攻擊
                elif event.key == pygame.K_p: # 換個暫停鍵
                    paused = not paused
                elif event.key in (pygame.K_g, pygame.K_4):
                    action_throw_grenade = 1.0
                elif event.key in (pygame.K_TAB, pygame.K_2):
                    action_switch_weapon = 1.0
                elif event.key in (pygame.K_h, pygame.K_3):
                    action_use_medkit = 1.0
                elif event.key == pygame.K_z:
                    action_drop_weapon = 1.0
                elif event.key == pygame.K_x:
                    action_drop_medkit = 1.0
                elif event.key == pygame.K_c:
                    action_drop_grenade = 1.0
                elif event.key == pygame.K_v:
                    action_drop_ammo = 1.0
        
        if paused or done:
            draw_frame()
            clock.tick(FPS)
            if done:
                result_str = ("勝利" if last_info.get("ai_win")
                              else ("失敗" if last_info.get("ai_lost")
                                    else "平局"))
                end_msg = font_big.render(f"結束：{result_str}    [ESC] 離開", True, (255, 220, 50))
                screen.blit(end_msg, (WIDTH // 2 - end_msg.get_width() // 2, HEIGHT // 2 - end_msg.get_height() // 2))
                pygame.display.flip()
                
                waiting = True
                while waiting:
                    for event in pygame.event.get():
                        if event.type in (pygame.QUIT, ):
                            waiting = False
                        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                            waiting = False
                    clock.tick(10)
                pygame.quit()
                return
            continue

        keys = pygame.key.get_pressed()
        mouse_btns = pygame.mouse.get_pressed()

        # index:
        # 0=up, 1=down, 2=left, 3=right,
        # 4=cw, 5=ccw, 6=attack, 7=dash,
        # 8=switch_weapon, 9=use_medkit,
        # 10=throw_grenade, 11=focus,
        # 12=drop_weapon, 13=drop_medkit,
        # 14=drop_grenade, 15=drop_ammo
        act = [0.0] * 16
        if keys[pygame.K_w]: act[0] = 1.0
        if keys[pygame.K_s]: act[1] = 1.0
        if keys[pygame.K_a]: act[2] = 1.0
        if keys[pygame.K_d]: act[3] = 1.0
        
        if keys[pygame.K_e] or keys[pygame.K_RIGHT]: act[4] = 1.0
        if keys[pygame.K_q] or keys[pygame.K_LEFT]: act[5] = 1.0
        
        if mouse_btns[0] or keys[pygame.K_RETURN]: act[6] = 1.0
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: act[7] = 1.0
        
        if action_switch_weapon > 0:
            act[8] = 1.0
            action_switch_weapon = 0.0
            
        if action_use_medkit > 0:
            act[9] = 1.0
            action_use_medkit = 0.0
            
        if action_throw_grenade > 0:
            act[10] = 1.0
            action_throw_grenade = 0.0
            
        if keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]: act[11] = 1.0

        if action_drop_weapon > 0:
            act[12] = 1.0
            action_drop_weapon = 0.0
        if action_drop_medkit > 0:
            act[13] = 1.0
            action_drop_medkit = 0.0
        if action_drop_grenade > 0:
            act[14] = 1.0
            action_drop_grenade = 0.0
        if action_drop_ammo > 0:
            act[15] = 1.0
            action_drop_ammo = 0.0

        # 將玩家動作放進 actions list
        actions = [act]
        # 如果有多個 AI 槽位，其他則送空動作 (全0) 讓他們發呆
        for _ in range(n_ai - 1):
            actions.append([0.0] * 16)

        # 這裡的 step 會根據 n_ai 數量返回對應格式
        result = env.step(actions, frame_skip=1)
        if env.n_learning_agents == 1:
            state_out, last_reward, done, last_info = result
        else:
            states_out, rewards_out, done, last_info = result
            # 讓畫面上的 last_reward 顯示我們控制的 0 號 AI 的成績
            last_reward = rewards_out[0]
            
        step += 1

        draw_frame()
        clock.tick(FPS)

def main():
    parser = argparse.ArgumentParser(description="親自進遊戲體驗與 Debug 的腳本")
    parser.add_argument("--stage", type=int, default=0,
                        choices=list(range(7)),
                        help="遊玩的階段 (0–6)")
    parser.add_argument("--n_ai", type=int, default=1,
                        help="遊戲中總共有幾個 AI 槽位 (例如 stage 1 預設需要 2，你可以指定)")
    parser.add_argument("--max_frames", type=int, default=5400,
                        help="遊戲最大幀數上限")
    parser.add_argument("--weapon", type=str, default=None,
                        choices=["pistol", "rifle", "shotgun", "sniper"],
                        help="起始武器")
    args = parser.parse_args()

    play_game(
        stage_id=args.stage,
        n_ai=args.n_ai,
        max_frames=args.max_frames,
        weapon_name=args.weapon
    )

if __name__ == "__main__":
    main()

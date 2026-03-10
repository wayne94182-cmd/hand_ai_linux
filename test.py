import torch
import time
import numpy as np
from game.config import VIEW_SIZE, ROWS, COLS
from gpu_renderer import GPURenderer

def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"測試裝置: {device}\n")

    # 模擬 512 步，192 個 Agent (96 envs * 2 AI)
    ROLLOUT_STEPS = 512
    FLAT_BATCH = 192
    TOTAL_SAMPLES = ROLLOUT_STEPS * FLAT_BATCH  # 98,304 張圖

    print(f"模擬資料量: {ROLLOUT_STEPS} 步 x {FLAT_BATCH} AI = {TOTAL_SAMPLES} 筆")

    # ─── 1. 產生模擬資料 ───
    # CPU 模式的圖片陣列 (10 channels, 15x15)
    cpu_images = torch.randn(TOTAL_SAMPLES, 10, VIEW_SIZE, VIEW_SIZE, dtype=torch.float32)
    
    # 定義各項實體的最大數量 (對齊 train.py)
    MAX_ALLIES = 3
    MAX_ENEMIES = 8
    MAX_ITEMS = 20
    MAX_THREATS = 32
    MAX_SOUNDS = 16

    # 混合模式的座標陣列 (正確形狀的空張量)
    cpu_agent_poses = torch.randn(TOTAL_SAMPLES, 4, dtype=torch.float32)
    cpu_grids = torch.randint(0, 2, (TOTAL_SAMPLES, ROWS, COLS), dtype=torch.float32)

    gpu_ally_poses = torch.zeros(TOTAL_SAMPLES, MAX_ALLIES, 3, device=device)
    gpu_ally_mask = torch.zeros(TOTAL_SAMPLES, MAX_ALLIES, dtype=torch.bool, device=device)

    gpu_enemy_poses = torch.zeros(TOTAL_SAMPLES, MAX_ENEMIES, 3, device=device)
    gpu_enemy_mask = torch.zeros(TOTAL_SAMPLES, MAX_ENEMIES, dtype=torch.bool, device=device)

    gpu_item_poses = torch.zeros(TOTAL_SAMPLES, MAX_ITEMS, 3, device=device)
    gpu_item_mask = torch.zeros(TOTAL_SAMPLES, MAX_ITEMS, dtype=torch.bool, device=device)

    gpu_threat_poses = torch.zeros(TOTAL_SAMPLES, MAX_THREATS, 3, device=device)
    gpu_threat_mask = torch.zeros(TOTAL_SAMPLES, MAX_THREATS, dtype=torch.bool, device=device)

    gpu_sound_waves = torch.zeros(TOTAL_SAMPLES, MAX_SOUNDS, 4, device=device)
    gpu_sound_mask = torch.zeros(TOTAL_SAMPLES, MAX_SOUNDS, dtype=torch.bool, device=device)

    gpu_poison = torch.zeros(TOTAL_SAMPLES, 4, device=device)
    
    torch.cuda.synchronize()

    # ─── 測試 A：傳送圖片 (模擬 CPU 模式) ───
    t0 = time.perf_counter()
    gpu_images = cpu_images.to(device, non_blocking=True)
    torch.cuda.synchronize()
    time_img_transfer = time.perf_counter() - t0
    print(f"[CPU 模式] 傳送所有圖片到 GPU 耗時: {time_img_transfer:.4f} 秒")

    # ─── 測試 B：傳送座標 (模擬混合模式) ───
    t0 = time.perf_counter()
    gpu_agent_poses = cpu_agent_poses.to(device, non_blocking=True)
    gpu_grids = cpu_grids.to(device, non_blocking=True)
    torch.cuda.synchronize()
    time_coord_transfer = time.perf_counter() - t0
    print(f"[混合模式] 傳送座標到 GPU 耗時:     {time_coord_transfer:.4f} 秒")

    # ─── 測試 C：GPU 分塊批次渲染 (真實混合模式的作法) ───
    renderer = GPURenderer(map_rows=ROWS, map_cols=COLS)
    t0 = time.perf_counter()
    
    CHUNK_SIZE = 1024  # 每次畫 1024 張，安全又快速，不會爆 VRAM
    for i in range(0, TOTAL_SAMPLES, CHUNK_SIZE):
        end = min(i + CHUNK_SIZE, TOTAL_SAMPLES)
        # 讓 GPU 批次消化，傳入對應形狀的 Mask
        rendered_imgs = renderer.render_batch(
            gpu_agent_poses[i:end], 
            gpu_ally_poses[i:end], gpu_ally_mask[i:end], 
            gpu_enemy_poses[i:end], gpu_enemy_mask[i:end], 
            gpu_item_poses[i:end], gpu_item_mask[i:end], 
            gpu_threat_poses[i:end], gpu_threat_mask[i:end], 
            gpu_sound_waves[i:end], gpu_sound_mask[i:end], 
            gpu_grids[i:end], gpu_poison[i:end], device=device
        )
    torch.cuda.synchronize()
    time_gpu_render = time.perf_counter() - t0
    print(f"[混合模式] GPU 分塊重畫十萬張圖片耗時: {time_gpu_render:.4f} 秒")

    # ─── 總結 ───
    print("\n--- 結論對比 ---")
    print(f"傳統 CPU 模式傳輸準備總耗時: {time_img_transfer:.4f} 秒 (且嚴重佔用系統 RAM)")
    print(f"混合模式 (傳座標 + GPU重畫): {(time_coord_transfer + time_gpu_render):.4f} 秒 (系統 RAM 幾近於 0)")

if __name__ == "__main__":
    run_benchmark()
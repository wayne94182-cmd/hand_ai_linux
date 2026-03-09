#!/usr/bin/env python3
"""
benchmark.py - 性能基準測試腳本
測試優化前後的訓練吞吐量
"""
import time
import torch
import numpy as np
from game import GameEnv
from ai import ConvLSTM, TeamPoolingCritic

def benchmark_env(num_steps=100, n_ai=2):
    """測試環境步進速度"""
    print(f"\n{'='*60}")
    print(f"環境步進基準測試（{num_steps} 步，{n_ai} AI）")
    print(f"{'='*60}")

    env = GameEnv(render_mode=False, stage_id=1, n_learning_agents=n_ai)
    env.reset()

    start = time.time()
    for _ in range(num_steps):
        actions = [[0.5] * 16 for _ in range(n_ai)]
        env.step(actions, frame_skip=2)
    elapsed = time.time() - start

    print(f"總時間: {elapsed:.2f} 秒")
    print(f"平均每步: {elapsed/num_steps*1000:.2f} ms")
    print(f"吞吐量: {num_steps/elapsed:.2f} steps/s")
    return elapsed

def benchmark_view_computation(num_calls=100):
    """測試視野計算速度"""
    print(f"\n{'='*60}")
    print(f"視野計算基準測試（{num_calls} 次調用）")
    print(f"{'='*60}")

    env = GameEnv(render_mode=False, stage_id=1, n_learning_agents=2)
    env.reset()

    start = time.time()
    for _ in range(num_calls):
        for agent in env.learning_agents:
            env._get_local_view(agent)
    elapsed = time.time() - start

    print(f"總時間: {elapsed:.2f} 秒")
    print(f"平均每次: {elapsed/num_calls*1000:.2f} ms")
    print(f"吞吐量: {num_calls/elapsed:.2f} calls/s")
    return elapsed

def benchmark_sound_rendering(num_calls=100):
    """測試聲音渲染速度"""
    print(f"\n{'='*60}")
    print(f"聲音渲染基準測試（{num_calls} 次調用）")
    print(f"{'='*60}")

    from game.audio import render_sound_channel, create_gunshot_wave

    # 創建一些波紋
    waves = [create_gunshot_wave(100 + i*10, 100 + i*10, i*5) for i in range(10)]

    start = time.time()
    for _ in range(num_calls):
        render_sound_channel(
            400, 400, 1.0, 0.0, 0.0, 1.0,
            waves, current_frame=50, prev_frame=48
        )
    elapsed = time.time() - start

    print(f"總時間: {elapsed:.2f} 秒")
    print(f"平均每次: {elapsed/num_calls*1000:.2f} ms")
    print(f"吞吐量: {num_calls/elapsed:.2f} calls/s")
    return elapsed

def benchmark_trajectory_storage(num_iterations=1000, batch_size=128):
    """測試軌跡存儲速度（預分配 vs 動態）"""
    print(f"\n{'='*60}")
    print(f"軌跡存儲基準測試（{num_iterations} 次，batch={batch_size}）")
    print(f"{'='*60}")

    # 預分配版本
    buf = np.zeros((num_iterations, batch_size, 10, 15, 15), dtype=np.float32)
    data = np.random.randn(10, 15, 15).astype(np.float32)

    start = time.time()
    for step in range(num_iterations):
        for idx in range(batch_size):
            buf[step, idx] = data
    elapsed_prealloc = time.time() - start

    print(f"預分配版本: {elapsed_prealloc:.3f} 秒")
    print(f"吞吐量: {num_iterations*batch_size/elapsed_prealloc:.0f} writes/s")

    return elapsed_prealloc

def benchmark_tensor_creation():
    """測試 torch.tensor vs torch.as_tensor"""
    print(f"\n{'='*60}")
    print(f"張量創建基準測試")
    print(f"{'='*60}")

    data = np.random.randn(100, 128, 10, 15, 15).astype(np.float32)

    # torch.tensor（拷貝）
    start = time.time()
    for _ in range(100):
        _ = torch.tensor(data)
    elapsed_copy = time.time() - start
    print(f"torch.tensor（拷貝）: {elapsed_copy:.3f} 秒")

    # torch.as_tensor（零拷貝）
    start = time.time()
    for _ in range(100):
        _ = torch.as_tensor(data)
    elapsed_zerocopy = time.time() - start
    print(f"torch.as_tensor（零拷貝）: {elapsed_zerocopy:.3f} 秒")

    speedup = elapsed_copy / elapsed_zerocopy
    print(f"加速比: {speedup:.2f}x")

    return speedup

if __name__ == "__main__":
    print("\n" + "="*60)
    print("訓練效率優化 - 性能基準測試")
    print("="*60)

    # 檢測 GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n設備: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # 運行基準測試
    try:
        benchmark_env(num_steps=50, n_ai=2)
        benchmark_view_computation(num_calls=100)
        benchmark_sound_rendering(num_calls=100)
        benchmark_trajectory_storage(num_iterations=500, batch_size=128)
        benchmark_tensor_creation()

        print(f"\n{'='*60}")
        print("✅ 所有基準測試完成")
        print("="*60)

    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()

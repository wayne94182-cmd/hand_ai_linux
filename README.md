# AI_madein_vibe 🤖

這個專案是一個基於 **PPO (Proximal Policy Optimization)** 與 **GRU (Gated Recurrent Unit)** 的強化學習 AI 戰鬥環境。具備視角受限的 **Egocentric FOV (第一人稱視野)** 與向量化運算優化。

## ✨ 特色功能
- **局部視野感知 (Egocentric FOV)**: AI 只能看到前方 100 度內、14 格距離的景象，且會受牆壁遮擋。
- **神經網路架構**: 使用 **CNN + GRU** 處理 POMDP（部分可觀察馬可夫決策過程）問題。
- **向量化 PPO 更新**: 全面優化訓練迴圈，大幅提升 GPU 到 52% 使用率。
- **Focus/ADS 模式**: AI 可以切換轉速模式進行精準瞄準。
- **高幀率模擬**: 60 FPS 環境，透過 Frame Skip 加速訓練。

## 🚀 快速開始
### 1. 安裝環境
建議使用 Python 3.12+ 與 PyTorch (含 CUDA 支援)。
```bash
python3 -m venv venv
source venv/bin/activate
pip install torch pygame numpy tqdm
```

### 2. 開始訓練
```bash
python train.py
```

### 3. 觀看 AI 表現
```bash
python watch.py [episode_number]
```

## 🛠️ 專案架構
- `game.py`: 核心遊戲邏輯、FOV 計算與渲染。
- `ai.py`: 模型定義 (CNN + GRU)。
- `train.py`: PPO 訓練腳本、BPTT 實現。
- `watch.py`: 模型載入與觀戰工具。

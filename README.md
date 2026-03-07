# Hand AI — Tactical Multi-Agent RL Environment 🤖🔫

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Pygame](https://img.shields.io/badge/Pygame-2.0%2B-green)](https://www.pygame.org/)

這是一個仿射擊遊戲架構的 **多代理人強化學習 (Multi-Agent RL)** 戰鬥環境。AI 透過視角受限的 6 通道視覺與 22 個感官純量輸入，在不斷變化的戰場中學習掩體利用、偵查、團隊協作與資源管理。

---

## 🚀 核心設計

### 🧠 神經網絡架構 (ConvLSTM + Attention)
- **視覺處理 (CNN)**: 處理 15×15 的 **Egocentric FOV** 視覺張量 (6 Channels)。
- **時序記憶 (LSTM)**: 使用 **256 維 LSTM** 作為核心，處理部分可觀察 (POMDP) 的戰場狀態。
- **團隊通訊 (Cross-Attention)**: 利用 Attention 機制接收動態數量的隊友通訊向量 (4-dim Continuous Comm)，實現情報共享。
- **動作空間**: 12 維離散動作頭 (Bernoulli)，支援同時執行多個操作（如：左轉 + 開火 + 打藥）。

### 🎮 戰鬥系統與要素
- **武裝系統**: 支援手槍、步槍、散彈槍、狙擊槍。每種武器具備不同的射程、擴散、裝彈速度。
- **道具管理**: 包含醫療箱 (Medkit) 手動補血讀條、手榴彈 (Grenade) AOE 爆炸傷害。
- **聲音物理**: 整合聲學傳播 (Sound Wave)，腳步聲與槍聲會產生圓形波紋，AI 可從「聽覺通道」判斷敵人位置。
- **運動模式**: 支援 **Dash (閃避)**。透過消耗 HP 換取瞬間爆發速度，增加戰鬥深度。

### 🎓 課程學習 (Curriculum Learning)
- **多階段訓練 (Stage 0–6)**: 從基礎移動、打固定靶，演進到複雜地圖搜索、1v1、以及多人大亂鬥。
- **動態地圖**: 支援多種地圖規模，從小型診所到大型倉庫迷宮。

---

## 📉 輸入維度說明

| 類別 | 維度 | 內容說明 |
| :--- | :--- | :--- |
| **視覺 (6ch)** | `(6, 15, 15)` | 地形、敵人(LOS)、隊友(Global)、威脅(彈道)、聲音波紋、安全區 |
| **純量 (22)** | `(22,)` | 武器槽 1-hot、彈藥比、換彈/打藥進度、HP 比、道具數、隊友方位 |
| **通訊 (In)** | `(K, 4)` | 接收來自 K 個隊友的 4 維連續向量 (經由 Cross-Attention) |

---

## �️ 安裝與啟動

### 1. 安裝環境
建議在 Linux 環境下執行，並確保具備 Python 3.12+ 與 PyTorch。
```bash
python3 -m venv venv
source venv/bin/activate
pip install torch numpy pygame tqdm numba
```

### 2. 多代理人訓練 (MAPPO)
```bash
# 預設啟動 128 個並行環境，每個環境 2 個 AI (1v1 或 2v0)
python train.py --n_ai 2 --stage 5 --stage_eps 50000
```
- `--n_ai`: 每個環境的學習代理人數量。
- `--stage`: 指定訓練階段目錄。
- `--resume`: 從 checkpoint 繼續訓練。

### 3. 高性能觀戰 (Watch)
```bash
# 觀看並顯示 AI 內部視覺層與通訊狀態
python watch.py --ckpt final --stage 5 --ai_view --show_comm
```
- `[Space]`: 暫停 / 繼續
- `[]] / [[]`: 加速 / 減速 (0.25x – 4.0x)
- `--ai_view`: 側邊欄即時渲染 AI 的 6 通道視覺觀測。

---

## � 專案架構
- `ai/`: 神經網絡核心。
    - `actor.py`: `ConvLSTM` 定義 (Actor)。
    - `critic.py`: `TeamPoolingCritic` 定義。
    - `comm.py`: 通訊向量處理。
- `game/`: 物理環境與模擬。
    - `env.py`: `GameEnv` 主核心，處理 60 FPS 物理模擬。
    - `entities.py`: Agent/Projectile/Grenade 類別。
    - `items.py`: 武器與道具規格。
    - `fov.py`: Numba 加速的視線與視野計算。
    - `audio.py`: 聲音傳播邏輯。
- `train.py`: MAPPO 訓練腳本，支援同步/非同步更新。
- `watch.py`: 具備可視化功能的觀戰工具。

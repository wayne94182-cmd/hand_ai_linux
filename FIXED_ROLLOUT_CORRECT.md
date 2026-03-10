# Fixed-Length Rollout 正確實現（重大修復）

## 問題根源

之前的實現有**兩個致命錯誤**，破壞了 Fixed-Length Rollout 的核心邏輯：

### 錯誤 1：每 512 步強制重置環境與 LSTM

**錯誤代碼位置**：在 `for batch_ep in pbar:` 循環內部

```python
# ❌ 錯誤：每個 batch 都重置
for batch_ep in pbar:
    h = torch.zeros(...)  # 強制失憶
    c = torch.zeros(...)
    env_states = vec_env.reset()  # 強制重啟遊戲
```

**後果**：
- 遊戲設定 `max_frames = 1200` 或 `5400`，但每 512 步就被強制重啟
- AI 永遠看不到毒圈縮小（Stage 4-6）
- LSTM 無法跨 batch 保持記憶，破壞了連續決策
- AI 無法學習長期戰術

### 錯誤 2：統計半截 Episode 數據

**錯誤代碼位置**：rollout 結束後

```python
# ❌ 錯誤：把未完成的 episode 當作完成的
for j in range(NUM_ENVS):
    if not env_had_done:
        # 遊戲還在進行中，卻強行記錄統計
        completed_episodes.append((down_count, is_win))
```

**後果**：
- 遊戲明明打到第 500 步（還沒結束），卻被當作「完成的 episode」
- 統計數據顯示的是「512 步時的狀態」，不是「完整遊戲的結果」
- `d0/d1/d2` 分佈完全不準確

---

## 正確的 Fixed-Length Rollout 概念

Fixed-Length Rollout 的核心是：

1. **採樣固定步數（512）用於 PPO 更新** ✅
2. **環境跨 batch 持續運行** ✅（不是每 512 步重啟）
3. **LSTM 記憶跨 batch 持續** ✅（只在 episode done 時清空）
4. **只統計完整的 episode** ✅（不統計半截數據）

**類比**：
- 想像 AI 在玩一局 1200 步的遊戲
- 每 512 步，AI 暫停去「上課」（PPO 更新權重）
- 上完課後，AI **帶著記憶** 繼續剛才的遊戲（從第 513 步開始）
- 只有遊戲真正結束（done=True）時，才重置環境與清空記憶

---

## 修復內容

### 修復 1：環境與 LSTM 初始化移到循環外

**修改位置**：[train.py:311-325](train.py#L311-L325)

```python
# ✅ 正確：訓練開始前只初始化一次
vec_env.set_stage(current_stage)

# LSTM 隱藏狀態：跨 batch 持續（只在 episode done 時清空）
h = torch.zeros(1, FLAT_BATCH, HIDDEN_SIZE, device=device)
c = torch.zeros(1, FLAT_BATCH, HIDDEN_SIZE, device=device)
last_comm = np.zeros((FLAT_BATCH, NUM_COMM), dtype=np.float32)

# 環境狀態：跨 batch 持續（只在 episode done 時重置）
env_states = vec_env.reset()

# Action mask：跨 batch 持續
last_masks = np.ones((FLAT_BATCH, NUM_ACTIONS_DISCRETE), dtype=bool)

# 現在才開始 batch 循環
for batch_ep in pbar:
    # ... 收集 512 步數據
    # ... PPO 更新
    # 下一個 batch 繼續用同樣的 h, c, env_states
```

### 修復 2：刪除半截 Episode 統計

**刪除代碼**：整段 `# ── Rollout 結束統計：包含未完成的 episode ──`

```python
# ❌ 已刪除這段錯誤代碼
# for j in range(NUM_ENVS):
#     if not env_had_done:
#         completed_episodes.append((down_count, is_win))
```

### 保留：只在真正 done 時統計

**正確位置**：[train.py:478-484](train.py#L478-L484)

```python
# ✅ 正確：只在遊戲真正結束時統計
if dones[j]:
    total_episode_count += 1
    down_count = infos[j].get("down_count", 0)
    is_win = 1 if infos[j].get("ai_win", False) else 0
    completed_episodes.append((down_count, is_win))

    # 然後才重置環境與清空 LSTM
    env_states[j] = new_states[j]
    h[:, flat, :] = 0.0
    c[:, flat, :] = 0.0
```

---

## 修復後的行為

### 時間軸示例（Stage 2，max_frames=1500）

**環境 0 的生命週期**：

```
Step 0-512   (Batch 0): 遊戲開始 → 收集 512 步 → PPO 更新
                        LSTM (h,c) 保留，環境繼續

Step 513-1024 (Batch 1): 繼續同一局遊戲 → 收集 512 步 → PPO 更新
                        LSTM (h,c) 保留，環境繼續

Step 1025-1200 (Batch 2): 遊戲結束（達到 max_frames 或有一方全滅）
                        done=True → 記錄統計 (down_count=2, is_win=1)
                        重置環境 + 清空 LSTM

Step 1201-1712 (Batch 2): 新遊戲開始 → 收集剩餘步數 → PPO 更新
```

**關鍵特性**：
- 每個 episode 可以跨越多個 batch（1-3 個）
- LSTM 記憶在整個 episode 期間保持
- 統計數據只記錄**完整完成的 episode**

### 統計數據恢復正常

**修復前**（每個 batch 都有數據）：
```
Batch 0: completed_episodes=64 (都是半截數據)
Batch 1: completed_episodes=64 (都是半截數據)
```

**修復後**（只記錄完成的）：
```
Batch 0: completed_episodes=8  (8 個環境真正完成了 episode)
Batch 1: completed_episodes=12 (12 個環境真正完成了 episode)
Batch 2: completed_episodes=15 (15 個環境真正完成了 episode)
```

**統計含義**：
- `d0`: 完整遊戲中，擊倒 0 個 NPC 的比例
- `d1`: 完整遊戲中，擊倒 1 個 NPC 的比例
- `d2`: 完整遊戲中，擊倒 2 個 NPC 的比例
- `d3`: 完整遊戲中，擊倒 3 個 NPC 的比例（全勝）
- `win_rate`: 完整遊戲的勝率

---

## 性能與訓練影響

### 優勢保留

✅ **cuDNN benchmark**：形狀固定 `(512, FLAT_BATCH, ...)`
✅ **Fused optimizer**：批量更新參數
✅ **Channels Last**：改善記憶體訪問
✅ **消除 GPU 等待**：不用等所有環境 done

### 新增優勢

✅ **長期記憶**：LSTM 可以跨 batch 保持狀態
✅ **完整 Episode**：AI 能體驗完整的 1200-5400 步遊戲
✅ **學習長期戰術**：看到毒圈縮小、資源爭奪、團隊配合
✅ **準確統計**：d0/d1/d2 反映真實表現

### 訓練時間預估

**完成 episode 的頻率**：
- Stage 0-1（短 episode，~300 步）：每個 batch 約 64×(512/300) ≈ **109 個 episode**
- Stage 2-3（中 episode，~800 步）：每個 batch 約 64×(512/800) ≈ **41 個 episode**
- Stage 4-6（長 episode，~2000 步）：每個 batch 約 64×(512/2000) ≈ **16 個 episode**

**統計穩定性**：
- 滾動窗口保留 200 個 episode
- Stage 0-1：約 2 個 batch 的數據
- Stage 2-3：約 5 個 batch 的數據
- Stage 4-6：約 12 個 batch 的數據

---

## 驗證方法

### 檢查 Episode 長度

修復後，你應該能在日誌中看到：

```
2026-03-10 12:40:00 [INFO] 進度: Stage2-追獵期, eps=96, win_rate=0.000
2026-03-10 12:40:25 [INFO] 進度: Stage2-追獵期, eps=192, win_rate=0.000
2026-03-10 12:40:50 [INFO] 進度: Stage2-追獵期, eps=288, win_rate=0.125, downs=[d0:0.38 d1:0.38 d2:0.12 d3:0.12]
```

**關鍵指標**：
- `win_rate` 應該從 0.000 開始逐漸變化（不是每個 batch 都有數據）
- `downs` 分佈應該合理（不是 d3=1.00 或極端值）
- Episode 完成數量應該隨 stage 變化（短 episode 多，長 episode 少）

### 檢查 LSTM 連續性

如果你想驗證 LSTM 確實跨 batch 保持：
1. 在 batch 循環開始處加 `logger.info(f"h sum: {h.sum().item()}")`
2. 第一個 batch 應該是 0（初始化）
3. 後續 batch 應該是非零值（保留了記憶）

---

## 總結

這次修復解決了兩個根本性錯誤：

1. **環境與 LSTM 跨 batch 持續** → AI 能體驗完整的長遊戲
2. **只統計完整 episode** → 數據反映真實表現

現在的實現才是真正的 **Fixed-Length Rollout**：
- 固定 512 步用於 PPO 更新（GPU 效率）
- 環境無縫持續運行（完整 episode 體驗）
- LSTM 記憶跨 batch 保持（長期決策能力）
- 統計數據準確可信（完整遊戲結果）

AI 現在可以學習真正的長期戰術了！🎯

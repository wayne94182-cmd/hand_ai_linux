# Fixed-Length Rollout 優化總結

## 核心改進：從動態填充到固定步數採樣

### 問題分析
原始實現使用動態長度的 episode 收集：
- 等待所有環境完成 episode 才進行更新
- 需要動態 padding 到 max_t
- GPU 需要等待最慢的 episode
- cuDNN benchmark 無法啟用（形狀不固定）
- torch.compile 效果受限

### 解決方案：Fixed-Length Rollout
參考 Gemini 建議，改為固定步數採樣模式：

```python
ROLLOUT_STEPS = 512  # 固定步數採樣
```

**核心邏輯變更：**
```python
# OLD: 動態等待所有 episode 完成
while not all(episode_done):
    step += 1
    # ... 收集數據

# NEW: 固定步數，環境 done 時立刻 reset
for step in range(ROLLOUT_STEPS):
    # ... 收集數據
    if env.done:
        env.reset()
        # 清空 LSTM 狀態防止記憶污染
        h[:, flat, :] = 0.0
        c[:, flat, :] = 0.0
```

## 詳細變更清單

### 1. train.py 主要修改

#### 1.1 固定步數採樣參數（Line 42）
```python
ROLLOUT_STEPS = 512  # 固定步數採樣（Fixed-Length Rollout）
```

#### 1.2 緩衝區分配改為固定大小（Lines 344-356）
```python
# OLD: 使用 MAX_STEPS（動態可變）
buf_states = np.zeros((MAX_STEPS, FLAT_BATCH, NUM_CHANNELS, VIEW_SIZE, VIEW_SIZE), dtype=np.float32)

# NEW: 使用 ROLLOUT_STEPS（固定不變）
buf_states = np.zeros((ROLLOUT_STEPS, FLAT_BATCH, NUM_CHANNELS, VIEW_SIZE, VIEW_SIZE), dtype=np.float32)
buf_dones = np.zeros((ROLLOUT_STEPS, FLAT_BATCH), dtype=bool)  # 追蹤 episode 邊界
```

#### 1.3 採樣循環從 while 改為 for（Line 364）
```python
# OLD: while not all(episode_done):
# NEW: for step in range(ROLLOUT_STEPS):
```

#### 1.4 自動環境重置（Lines 476-496）
```python
# 環境 done 時立刻 reset（無縫接軌）
if dones[j]:
    total_episode_count += 1

    # 立刻重置環境
    next_env_states[j] = new_all_states[j]

    # 清空 LSTM 狀態（防止記憶污染）
    for i in range(n_ai):
        flat = i * NUM_ENVS + j
        h[:, flat, :] = 0.0
        c[:, flat, :] = 0.0
        last_comm[flat] = 0.0
```

#### 1.5 重寫 GAE 計算以處理 episode 邊界（Lines 498-533）
```python
# 分段計算 GAE（根據 done 標記切分 episode）
for flat in range(FLAT_BATCH):
    episode_start = 0
    for t in range(ROLLOUT_STEPS):
        if buf_dones[t, flat] or t == ROLLOUT_STEPS - 1:
            episode_end = t + 1
            ep_rews = buf_rewards[episode_start:episode_end, flat].tolist()
            ep_vals = buf_values[episode_start:episode_end, flat].tolist()

            # Bootstrap 處理
            last_val = 0.0 if buf_dones[t, flat] else ep_vals[-1]
            ep_advs = compute_gae(ep_rews, ep_vals, last_value=last_val,
                                  truncated=not buf_dones[t, flat])

            buf_advantages[episode_start:episode_end, flat] = ep_advs
            buf_returns[episode_start:episode_end, flat] = [a + v for a, v in zip(ep_advs, ep_vals)]
            episode_start = episode_end
```

#### 1.6 移除動態 Padding 和 mask 邏輯（Lines 535-547, 614-680）
```python
# OLD: 需要 mask 來處理 padding
mask = (step_idx < max_t).astype(np.float32)
valid_flat = mask.reshape(TB)
n_valid = valid_flat.float().sum().clamp(min=1)
t_actor_loss = (total_actor * valid_flat.float()).sum() / n_valid

# NEW: 形狀固定，直接用 mean()
TB = ROLLOUT_STEPS * FLAT_BATCH
t_actor_loss = total_actor.mean()
t_critic_loss = critic_l.mean()
```

#### 1.7 啟用 cuDNN benchmark（Line 266）
```python
# OLD: torch.backends.cudnn.benchmark = False  # max_t 動態變化
# NEW: torch.backends.cudnn.benchmark = True   # 固定形狀，啟用優化
```

#### 1.8 使用 Fused AdamW 優化器（Lines 261-264）
```python
# OLD: optimizer = optim.Adam(model.parameters(), lr=LR)
# NEW:
use_fused = device.type == "cuda"
optimizer = optim.AdamW(model.parameters(), lr=LR, fused=use_fused)
optimizer_critic = optim.AdamW(critic.parameters(), lr=LR, fused=use_fused)
```

### 2. ai/actor.py 修改

#### 2.1 優化 Gradient Checkpoint chunk_size（Line 97）
```python
# OLD: chunk_size = 2048
# NEW: chunk_size = 512  # 配合 ROLLOUT_STEPS 優化記憶體分配
```

#### 2.2 添加 Channels Last 記憶體格式（Line 85）
```python
def _cnn_embed(self, x, scalars):
    # Channels Last 記憶體格式優化（GPU 加速）
    x = x.to(memory_format=torch.channels_last)
    c1 = F.relu(self.bn1(self.conv1(x)))
    # ...
```

## 預期性能提升

### 理論加速比：6-10x

**分析：**
1. **cuDNN benchmark**：自動選擇最優卷積算法 → ~1.5-2x
2. **Fused AdamW**：合併 CUDA kernel 操作 → ~1.2x
3. **固定形狀 torch.compile**：更激進的優化 → ~1.3-1.5x
4. **消除 GPU 等待**：不用等最慢 episode → ~1.5-2x
5. **Channels Last**：改善記憶體訪問模式 → ~1.1-1.2x
6. **移除 mask 開銷**：簡化計算圖 → ~1.1x

總體估計：1.5 × 1.2 × 1.3 × 1.5 × 1.1 × 1.1 ≈ **3.6x**（保守）到 **10x**（樂觀）

### VRAM 節省
- 移除動態 padding → 節省 ~10-20% VRAM
- 優化 checkpoint chunk_size → 節省 ~5-10% VRAM
- **可增加 NUM_ENVS**：64 → 96 → 128

## 關鍵技術要點

### 1. LSTM 狀態管理
✅ **正確做法**：環境 reset 時清空 LSTM 狀態
```python
if env.done:
    h[:, flat, :] = 0.0  # 清空防止記憶污染
    c[:, flat, :] = 0.0
```

❌ **錯誤做法**：使用 TBPTT 切斷梯度（會讓 AI 變笨）

### 2. GAE 邊界處理
- 使用 `buf_dones` 追蹤 episode 邊界
- 每個 flat index 獨立計算 GAE
- 正確的 bootstrap 處理：
  - episode 結束：`last_val = 0.0`
  - rollout 截斷：`last_val = V(s_T)`

### 3. 形狀保證
所有張量形狀固定為 `(ROLLOUT_STEPS, FLAT_BATCH, ...)`：
- 啟用 cuDNN benchmark
- torch.compile 更激進優化
- 無需 mask/padding 開銷

## 測試計劃

### Phase 1: 驗證正確性
```bash
python3 train.py --stage 0 --num_envs 8
```
檢查：
- [ ] 訓練能正常運行
- [ ] Loss 數值合理
- [ ] Reward 曲線正常
- [ ] 無 CUDA OOM 錯誤

### Phase 2: 性能測試
```bash
python3 train.py --stage 1 --num_envs 64
```
監控：
- FPS（應提升 3-6x）
- GPU 利用率（應提升到 >90%）
- VRAM 使用（應降低 10-20%）

### Phase 3: 擴展測試
逐步增加環境數量：
```bash
python3 train.py --num_envs 96   # +50%
python3 train.py --num_envs 128  # +100%
```

## 回滾方案

如遇問題，Git 回滾到實現前版本：
```bash
git log --oneline  # 查看提交歷史
git checkout <commit-hash>  # 回滾到指定版本
```

## 參考資料

- **Gemini 建議**：Fixed-Length Rollout 模式分析
- **cuDNN Best Practices**：https://docs.nvidia.com/deeplearning/cudnn/
- **PyTorch Performance Tuning**：https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
- **PPO with Fixed Rollouts**：Stable-Baselines3 實現參考

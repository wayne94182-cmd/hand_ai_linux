# Stage 2+ 遊戲配置調整

## 修改內容

### 1. 彈匣上限提升：5 → 12

**檔案**：[game/entities.py](game/entities.py#L75)

```python
# OLD: self.max_ammo_boxes: int = 5
# NEW: self.max_ammo_boxes: int = 12
```

### 2. Stage 2+ 無初始武器

**檔案**：[game/env.py](game/env.py#L253-L265)

```python
# 出生武器：Stage 0-1 有 PISTOL，Stage 2+ 需要自己撿
if self.stage_id <= 1:
    a.weapon_slots = [PISTOL]
    a.active_slot = 0
    a.ammo = PISTOL.mag_size
    a.max_ammo = PISTOL.mag_size
    a.reload_delay = PISTOL.reload_frames
else:
    a.weapon_slots = []  # Stage 2+ 無初始武器
    a.active_slot = 0
    a.ammo = 0
    a.max_ammo = 0
    a.reload_delay = 0
```

## 各 Stage 配置總結

| Stage | 初始武器 | 初始彈匣 | 彈匣上限 |
|-------|---------|---------|---------|
| 0 (瞄準期) | PISTOL | 3 | 12 |
| 1 (打靶期) | PISTOL | 3 | 12 |
| 2 (追獵期) | **無** | 3 | 12 |
| 3 (生存期) | **無** | 3 | 12 |
| 4 (戰術期) | **無** | 3 | 12 |
| 5 (自我博弈) | **無** | 3 | 12 |
| 6 (多隊博弈) | **無** | 3 | 12 |

## 遊戲影響

### Stage 2+ 新增挑戰：
1. **必須先找武器**才能戰鬥
2. **初期脆弱**：遇到敵人只能逃跑
3. **資源搜尋**更重要：武器、彈匣、醫療包都需要撿取
4. **策略性提升**：需要權衡搜尋時間 vs 戰鬥時機

### 彈匣上限提升好處：
1. **持續戰鬥能力**增強：12 個彈匣足以支撐長時間戰鬥
2. **減少缺彈焦慮**：可以更積極地使用彈藥壓制敵人
3. **獎勵搜尋行為**：撿到彈匣盒更有價值

## 訓練影響

### AI 需要學習的新技能：
1. **搜尋行為**：優先尋找武器
2. **無武器脫戰**：沒武器時避免戰鬥
3. **資源管理**：平衡彈匣、醫療包、手榴彈的攜帶
4. **時機判斷**：何時搜尋 vs 何時戰鬥

### 預期訓練時間：
- Stage 2 初期可能會出現大量"無武器死亡"
- 需要 ~10k-20k episodes 學會"先找武器再戰鬥"
- 完全掌握資源搜尋策略可能需要 50k+ episodes

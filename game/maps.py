import numpy as np
import os

from game.config import ROWS, COLS


def create_base_grid():
    m = np.zeros((ROWS, COLS), dtype=np.int8)
    m[0, :] = 1
    m[-1, :] = 1
    m[:, 0] = 1
    m[:, -1] = 1
    return m


MAP_TINY_1 = np.ones((ROWS, COLS), dtype=np.int8)
MAP_TINY_1[6:18, 8:24] = 0
MAP_TINY_1[11:13, 14:16] = 1
MAP_TINY_1[11, 17:19] = 1

MAP_MEDIUM_BASIC = np.array(
    [
        [1] * 32,
        [1] + [0] * 30 + [1],
        [1] + [0] * 30 + [1],
        [1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1] + [0] * 30 + [1],
        [1] + [0] * 30 + [1],
        [1] + [0] * 30 + [1],
        [1] + [0] * 30 + [1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1],
        [1] + [0] * 30 + [1],
        [1] + [0] * 30 + [1],
        [1] + [0] * 30 + [1],
        [1] * 32,
    ],
    dtype=np.int8,
)

MAP_SMALL_3 = np.ones((ROWS, COLS), dtype=np.int8)
MAP_SMALL_3[4:14, 4:14] = 0
MAP_SMALL_3[10:20, 18:28] = 0
MAP_SMALL_3[10:14, 14:18] = 0

MAP_MEDIUM_1 = create_base_grid()
MAP_MEDIUM_1[5:9, 7:11] = 1
MAP_MEDIUM_1[5:9, 21:25] = 1
MAP_MEDIUM_1[15:19, 7:11] = 1
MAP_MEDIUM_1[15:19, 21:25] = 1

MAP_MEDIUM_2 = create_base_grid()
MAP_MEDIUM_2[11:13, 6:14] = 1
MAP_MEDIUM_2[11:13, 18:26] = 1
MAP_MEDIUM_2[6:10, 15:17] = 1
MAP_MEDIUM_2[14:18, 15:17] = 1

# --- 恢復與新增地圖 (確保實心) ---

# 1. 辦公室風格 2
MAP_MEDIUM_OFFICE_2 = create_base_grid()
MAP_MEDIUM_OFFICE_2[2:10, 8] = 1
MAP_MEDIUM_OFFICE_2[14:22, 23] = 1
MAP_MEDIUM_OFFICE_2[11, 2:12] = 1
MAP_MEDIUM_OFFICE_2[11, 20:30] = 1
MAP_MEDIUM_OFFICE_2[5:7, 15:17] = 1
MAP_MEDIUM_OFFICE_2[18:20, 15:17] = 1

# 2. 街區交火
MAP_MEDIUM_STREETS = create_base_grid()
MAP_MEDIUM_STREETS[2:10, [8, 16, 24]] = 1
MAP_MEDIUM_STREETS[14:22, [8, 16, 24]] = 1
MAP_MEDIUM_STREETS[11, 4:28] = 1
MAP_MEDIUM_STREETS[11, 14:18] = 0

# 3. 柱陣
MAP_MEDIUM_PILLARS = create_base_grid()
# 使用 meshgrid 與 broadcasting 產生實心方塊陣列
rr, cc = np.meshgrid(np.arange(4, 21, 5), np.arange(6, 27, 7), indexing='ij')
R = rr[..., None, None] + np.arange(2)[:, None]
C = cc[..., None, None] + np.arange(3)
MAP_MEDIUM_PILLARS[R, C] = 1

# 4. 舊有競技場地圖恢復
MAP_MEDIUM_CROSSFIRE = create_base_grid()
idx = np.arange(5, 20)
MAP_MEDIUM_CROSSFIRE[idx, idx + 4] = 1
MAP_MEDIUM_CROSSFIRE[idx, COLS - idx - 5] = 1
MAP_MEDIUM_CROSSFIRE[11:13, :] = 0
MAP_MEDIUM_CROSSFIRE[:, 15:17] = 0
# 補回外牆
MAP_MEDIUM_CROSSFIRE[0, :] = 1
MAP_MEDIUM_CROSSFIRE[-1, :] = 1
MAP_MEDIUM_CROSSFIRE[:, 0] = 1
MAP_MEDIUM_CROSSFIRE[:, -1] = 1

MAP_MEDIUM_OFFICE = create_base_grid()
MAP_MEDIUM_OFFICE[6, 1:12] = 1
MAP_MEDIUM_OFFICE[6, 15:26] = 1
MAP_MEDIUM_OFFICE[17, 6:20] = 1
MAP_MEDIUM_OFFICE[17, 24:31] = 1
MAP_MEDIUM_OFFICE[1:7, 12] = 1
MAP_MEDIUM_OFFICE[11:18, 12] = 1
MAP_MEDIUM_OFFICE[6:12, 20] = 1
MAP_MEDIUM_OFFICE[17:23, 20] = 1
# 填滿左上角的封閉空間
MAP_MEDIUM_OFFICE[1:6, 1:12] = 1

MAP_MEDIUM_ISLANDS = create_base_grid()
# 島嶼定義：(r, c, h, w)
# 由於島嶼尺寸不一，改用個別切片賦值以維持效能且無 loop
MAP_MEDIUM_ISLANDS[4:7, 4:7] = 1
MAP_MEDIUM_ISLANDS[4:7, 25:28] = 1
MAP_MEDIUM_ISLANDS[17:20, 4:7] = 1
MAP_MEDIUM_ISLANDS[17:20, 25:28] = 1
MAP_MEDIUM_ISLANDS[10:14, 10:12] = 1
MAP_MEDIUM_ISLANDS[10:14, 20:22] = 1
MAP_MEDIUM_ISLANDS[6:8, 14:18] = 1
MAP_MEDIUM_ISLANDS[16:18, 14:18] = 1


MAP_MEDIUM_BRAWL_SHOOTING = create_base_grid()
pts_r = np.array([4, 18, 8, 14, 2, 20]).reshape(-1, 1, 1)
pts_c = np.array([12, 18, 6, 24, 2, 28]).reshape(-1, 1, 1)
R = pts_r + np.arange(2).reshape(1, 2, 1)
C = pts_c + np.arange(4)
MAP_MEDIUM_BRAWL_SHOOTING[R, C] = 1
MAP_MEDIUM_BRAWL_SHOOTING[10:14, 14:18] = 1

MAP_MEDIUM_BRAWL_SNAKE = create_base_grid()
MAP_MEDIUM_BRAWL_SNAKE[5:10, 6:8] = 1
MAP_MEDIUM_BRAWL_SNAKE[14:19, 24:26] = 1
pts_r = np.array([4, 18, 8, 14, 11, 5, 17]).reshape(-1, 1, 1)
pts_c = np.array([15, 12, 22, 8, 15, 25, 5]).reshape(-1, 1, 1)
R = pts_r + np.arange(2).reshape(1, 2, 1)
C = pts_c + np.arange(2)
MAP_MEDIUM_BRAWL_SNAKE[R, C] = 1

MAP_MEDIUM_BRAWL_ARENA = create_base_grid()
MAP_MEDIUM_BRAWL_ARENA[[6, 16], 8:24] = 1
MAP_MEDIUM_BRAWL_ARENA[6:18, [10, 21]] = 1
MAP_MEDIUM_BRAWL_ARENA[6, 15:17] = 0
MAP_MEDIUM_BRAWL_ARENA[16, 15:17] = 0
MAP_MEDIUM_BRAWL_ARENA[11:13, 10] = 0
MAP_MEDIUM_BRAWL_ARENA[11:13, 21] = 0

MAP_MEDIUM_BRAWL_STRAIGHT = create_base_grid()
rows = np.arange(4, 20, 4).reshape(-1, 1) + np.arange(2)
MAP_MEDIUM_BRAWL_STRAIGHT[rows, 6] = 1
MAP_MEDIUM_BRAWL_STRAIGHT[rows, 25] = 1
MAP_MEDIUM_BRAWL_STRAIGHT[11:13, 2:8] = 1
MAP_MEDIUM_BRAWL_STRAIGHT[11:13, 24:30] = 1
MAP_MEDIUM_BRAWL_STRAIGHT[11:13, 14:18] = 1

# --- 新增的小尺寸地圖 ---

MAP_SMALL_CLINIC = np.ones((ROWS, COLS), dtype=np.int8)
MAP_SMALL_CLINIC[4:20, 6:26] = 0
MAP_SMALL_CLINIC[4:10, 15:17] = 1
MAP_SMALL_CLINIC[14:20, 15:17] = 1
MAP_SMALL_CLINIC[11:13, 10:14] = 1
MAP_SMALL_CLINIC[11:13, 18:22] = 1

MAP_SMALL_OFFICE_3 = np.ones((ROWS, COLS), dtype=np.int8)
MAP_SMALL_OFFICE_3[5:19, 8:24] = 0
MAP_SMALL_OFFICE_3[5:8, 12:14] = 1
MAP_SMALL_OFFICE_3[16:19, 18:20] = 1
MAP_SMALL_OFFICE_3[11:13, 11:13] = 1
MAP_SMALL_OFFICE_3[11:13, 19:21] = 1

MAP_SMALL_STREET_2 = np.ones((ROWS, COLS), dtype=np.int8)
MAP_SMALL_STREET_2[6:18, 4:28] = 0
MAP_SMALL_STREET_2[6:10, 10:12] = 1
MAP_SMALL_STREET_2[14:18, 10:12] = 1
MAP_SMALL_STREET_2[6:10, 20:22] = 1
MAP_SMALL_STREET_2[14:18, 20:22] = 1
MAP_SMALL_STREET_2[11, 4:9] = 1
MAP_SMALL_STREET_2[11, 23:28] = 1

MAP_SMALL_LAB = np.ones((ROWS, COLS), dtype=np.int8)
MAP_SMALL_LAB[3:21, 5:27] = 0
MAP_SMALL_LAB[3:8, 10] = 1
MAP_SMALL_LAB[3:8, 21] = 1
MAP_SMALL_LAB[16:21, 10] = 1
MAP_SMALL_LAB[16:21, 21] = 1
MAP_SMALL_LAB[11:13, 5:12] = 1
MAP_SMALL_LAB[11:13, 20:27] = 1

MAP_SMALL_CAFE = np.ones((ROWS, COLS), dtype=np.int8)
MAP_SMALL_CAFE[7:17, 7:25] = 0
pts_r = np.array([9, 9, 14, 14, 11]).reshape(-1, 1, 1)
pts_c = np.array([10, 21, 10, 21, 15]).reshape(-1, 1, 1)
R = pts_r + np.arange(2).reshape(1, 2, 1)
C = pts_c + np.arange(2)
MAP_SMALL_CAFE[R, C] = 1

# --- 新增大尺寸 96x96 地圖 ---
def load_large_map(filename):
    path = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(path):
        return np.load(path)
    # 預防檔案遺失，產生個帶邊界的空陣列
    m = np.zeros((96, 96), dtype=np.int8)
    m[0, :] = 1
    m[-1, :] = 1
    m[:, 0] = 1
    m[:, -1] = 1
    return m

MAP_LARGE_LAKE = load_large_map("brawlstars_中央湖泊.npy")
MAP_LARGE_STREAM = load_large_map("brawlstars_幽暗溪流.npy")
MAP_LARGE_MAZE = load_large_map("brawlstars_炎熱迷宮.npy")
MAP_LARGE_CIRCLE = load_large_map("brawlstars_環形地帶.npy")
MAP_LARGE_STONE = load_large_map("brawlstars_石牆亂鬥.npy")

# 更新 ALL_MAPS
ALL_MAPS = {
    # 原始地圖
    "MAP_TINY_1": MAP_TINY_1,
    "MAP_MEDIUM_BASIC": MAP_MEDIUM_BASIC,
    "MAP_SMALL_3": MAP_SMALL_3,
    "MAP_MEDIUM_1": MAP_MEDIUM_1,
    "MAP_MEDIUM_2": MAP_MEDIUM_2,
    # 競技場地圖
    "MAP_MEDIUM_CROSSFIRE": MAP_MEDIUM_CROSSFIRE,
    "MAP_MEDIUM_OFFICE": MAP_MEDIUM_OFFICE,
    "MAP_MEDIUM_OFFICE_2": MAP_MEDIUM_OFFICE_2,
    "MAP_MEDIUM_STREETS": MAP_MEDIUM_STREETS,
    "MAP_MEDIUM_PILLARS": MAP_MEDIUM_PILLARS,
    "MAP_MEDIUM_ISLANDS": MAP_MEDIUM_ISLANDS,
    # 荒野風格
    "MAP_MEDIUM_BRAWL_SHOOTING": MAP_MEDIUM_BRAWL_SHOOTING,
    "MAP_MEDIUM_BRAWL_SNAKE": MAP_MEDIUM_BRAWL_SNAKE,
    "MAP_MEDIUM_BRAWL_ARENA": MAP_MEDIUM_BRAWL_ARENA,
    "MAP_MEDIUM_BRAWL_STRAIGHT": MAP_MEDIUM_BRAWL_STRAIGHT,
    # 新小尺寸地圖
    "MAP_SMALL_CLINIC": MAP_SMALL_CLINIC,
    "MAP_SMALL_OFFICE_3": MAP_SMALL_OFFICE_3,
    "MAP_SMALL_STREET_2": MAP_SMALL_STREET_2,
    "MAP_SMALL_LAB": MAP_SMALL_LAB,
    "MAP_SMALL_CAFE": MAP_SMALL_CAFE,
    # 新大尺寸地圖
    "MAP_LARGE_LAKE": MAP_LARGE_LAKE,
    "MAP_LARGE_STREAM": MAP_LARGE_STREAM,
    "MAP_LARGE_MAZE": MAP_LARGE_MAZE,
    "MAP_LARGE_CIRCLE": MAP_LARGE_CIRCLE,
    "MAP_LARGE_STONE": MAP_LARGE_STONE,
}

MAPS = list(ALL_MAPS.values())

SMALL_MAPS = [
    ALL_MAPS["MAP_TINY_1"],
    ALL_MAPS["MAP_SMALL_3"],
    ALL_MAPS["MAP_SMALL_CLINIC"],
    ALL_MAPS["MAP_SMALL_OFFICE_3"],
    ALL_MAPS["MAP_SMALL_STREET_2"],
    ALL_MAPS["MAP_SMALL_LAB"],
    ALL_MAPS["MAP_SMALL_CAFE"],
]

MEDIUM_MAPS = [
    ALL_MAPS["MAP_MEDIUM_BASIC"],
    ALL_MAPS["MAP_MEDIUM_1"],
    ALL_MAPS["MAP_MEDIUM_2"],
    ALL_MAPS["MAP_MEDIUM_CROSSFIRE"],
    ALL_MAPS["MAP_MEDIUM_OFFICE"],
    ALL_MAPS["MAP_MEDIUM_OFFICE_2"],
    ALL_MAPS["MAP_MEDIUM_STREETS"],
    ALL_MAPS["MAP_MEDIUM_PILLARS"],
    ALL_MAPS["MAP_MEDIUM_ISLANDS"],
    ALL_MAPS["MAP_MEDIUM_BRAWL_SHOOTING"],
    ALL_MAPS["MAP_MEDIUM_BRAWL_SNAKE"],
    ALL_MAPS["MAP_MEDIUM_BRAWL_ARENA"],
    ALL_MAPS["MAP_MEDIUM_BRAWL_STRAIGHT"],
]

LARGE_MAPS = [
    ALL_MAPS["MAP_LARGE_LAKE"],
    ALL_MAPS["MAP_LARGE_STREAM"],
    ALL_MAPS["MAP_LARGE_MAZE"],
    ALL_MAPS["MAP_LARGE_CIRCLE"],
    ALL_MAPS["MAP_LARGE_STONE"],
]

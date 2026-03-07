import numpy as np
import sys
import os

# 載入五個地圖 npy 檔
MAP_FILES = {
    "MAP_LARGE_LAKE": "game/brawlstars_中央湖泊.npy",
    "MAP_LARGE_STREAM": "game/brawlstars_幽暗溪流.npy",
    "MAP_LARGE_MAZE": "game/brawlstars_炎熱迷宮.npy",
    "MAP_LARGE_CIRCLE": "game/brawlstars_環形地帶.npy",
    "MAP_LARGE_STONE": "game/brawlstars_石牆亂鬥.npy"
}

ALL_96_MAPS = {}
for name, filename in MAP_FILES.items():
    if os.path.exists(filename):
        try:
            # 直接載入已經處理過（有牆壁、無密閉空間）的地圖檔
            ALL_96_MAPS[name] = np.load(filename)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    else:
        print(f"Warning: {filename} not found.")

if __name__ == "__main__":
    if not ALL_96_MAPS:
        print("No maps loaded. Check if .npy files exist.")
        sys.exit(1)
        
    # 如果沒指定參數，預設顯示第一個地圖
    selected_map = sys.argv[1] if len(sys.argv) > 1 else "MAP_LARGE_LAKE"
    
    if selected_map in ALL_96_MAPS:
        grid = ALL_96_MAPS[selected_map]
        print(f"Visualizing {selected_map} ({grid.shape[0]}x{grid.shape[1]}):")
        # 顯示地圖
        for r in range(grid.shape[0]):
            # 牆壁使用 #，空地使用空格
            row_str = "".join(["#" if grid[r, c] == 1 else " " for c in range(grid.shape[1])])
            print(row_str)
    else:
        print(f"Map '{selected_map}' not found.")
        print("Available maps:", ", ".join(ALL_96_MAPS.keys()))

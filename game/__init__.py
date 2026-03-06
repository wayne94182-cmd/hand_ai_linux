# game/__init__.py  — re-export everything so that
#   from game import GameEnv, MAPS, ...
# keeps working exactly as before.

from game.config import (
    GameConfig, StageSpec, STAGE_SPECS, get_stage_spec,
    TILE_SIZE, COLS, ROWS, WIDTH, HEIGHT, FPS,
    NUM_ACTIONS, MAX_FRAMES,
    VIEW_SIZE, VIEW_CENTER, FOV_DEGREES, HALF_FOV, VIEW_RANGE,
)

from game.maps import (
    ALL_MAPS, MAPS, SMALL_MAPS,
)

from game.entities import (
    Agent, Projectile, Grenade,
)

from game.items import (
    WeaponSpec, PISTOL, RIFLE, SHOTGUN, SNIPER, WEAPON_TYPES,
    GroundItem, try_auto_pickup,
)

from game.audio import SoundWave

from game.env import GameEnv, NUM_CHANNELS, NUM_SCALARS


if __name__ == "__main__":
    env = GameEnv(render_mode=False, stage_id=1)
    states = env.reset()
    view, sc = states[0] if isinstance(states, list) else states
    print("重構成功，view:", view.shape, "scalars:", sc.shape)

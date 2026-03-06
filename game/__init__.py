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

from game.env import (
    Agent, Projectile, GameEnv,
)


if __name__ == "__main__":
    from game import GameEnv, get_stage_spec, MAPS
    env = GameEnv(render_mode=False, stage_id=0)
    s = env.reset()
    print("重構成功，狀態形狀:", s[0].shape, s[1].shape)

from gymnasium.envs.registration import register
from .moving_firefighter import MovingFirefighter

register(
    id="mfp/MovingFirefighter-v0",
    entry_point="moving_firefighter_env:MovingFirefighter",
    max_episode_steps=300
)

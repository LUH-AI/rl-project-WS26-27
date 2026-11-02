"""Go2 velocity task variant: identical to the standard config
(velocity_env_cfg.RobotEnvCfg) except for one added reward term that locks
the gait to a pace (same-side legs move together) instead of whatever gait
the standard reward set converges to on its own. Everything else (terrain,
observations, actions, other rewards, terminations, curriculum) is
inherited unchanged.

Uses mdp.feet_gait, the same clock-based gait-locking reward already used
by the H1 biped config (robots/h1/velocity_env_cfg.py), applied here with a
four-leg offset pattern instead of H1's two-leg alternating one.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg import RewardsCfg as BaseRewardsCfg
from .velocity_env_cfg import RobotEnvCfg as BaseRobotEnvCfg

# Offsets assume the contact sensor resolves ".*_foot" bodies in the same
# order as this asset's joint_sdk_names (FR, FL, RR, RL) -
# unitree_rl_lab/assets/robots/unitree.py. This is the expected order given
# how consistently that ordering appears elsewhere in the asset definition,
# but it has not been confirmed against the sensor's actual resolved
# body_names at runtime. Print `sensor_cfg.body_names` (or
# `contact_sensor.body_names`) once before a real training run and reorder
# this list if it doesn't match FR, FL, RR, RL.
#
# Pace pairs same-side legs together (FR+RR in phase, FL+RL in phase),
# as opposed to trot's diagonal pairing.
PACE_OFFSETS = [0.0, 0.5, 0.0, 0.5]  # FR, FL, RR, RL

# Starting point, not a measured value - a quadruped's natural step period
# at the trained command speeds is unknown until a standard-policy rollout
# is inspected. Tune this (and PACE_THRESHOLD) if the gait looks rushed or
# dragging relative to the standard policy's own cadence.
PACE_PERIOD_S = 0.5
PACE_THRESHOLD = 0.5  # duty factor: fraction of each leg's cycle in stance

# Starting weight, not a validated value. air_time_variance (-1.0, inherited
# from the standard config) and feet_air_time already pull the standard
# policy toward its own emergent footfall pattern; this may need to be
# raised above the H1 precedent (0.5) for the pace schedule to actually win
# out rather than being absorbed without changing behavior - check this by
# rendering a rollout before trusting the result.
PACE_GAIT_WEIGHT = 1.0


@configclass
class RewardsCfg(BaseRewardsCfg):
    gait = RewTerm(
        func=mdp.feet_gait,
        weight=PACE_GAIT_WEIGHT,
        params={
            "period": PACE_PERIOD_S,
            "offset": PACE_OFFSETS,
            "threshold": PACE_THRESHOLD,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
        },
    )


@configclass
class RobotEnvCfg(BaseRobotEnvCfg):
    rewards: RewardsCfg = RewardsCfg()


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges

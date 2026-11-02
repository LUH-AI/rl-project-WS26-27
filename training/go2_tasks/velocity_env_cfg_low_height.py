"""Go2 velocity task variant: identical to the standard config
(velocity_env_cfg.RobotEnvCfg) except for one added reward term that biases
the policy toward a lower standing/walking height. Everything else
(terrain, observations, actions, other rewards, terminations, curriculum)
is inherited unchanged.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg import RewardsCfg as BaseRewardsCfg
from .velocity_env_cfg import RobotEnvCfg as BaseRobotEnvCfg

# Nominal spawn/standing height for this asset is 0.4 m
# (unitree_rl_lab/assets/robots/unitree.py: UNITREE_GO2_CFG.init_state.pos).
# This target is a starting point, not a validated value: check in sim that
# the legs still clear the ground and don't hit joint limits before
# training to convergence, and adjust if the crouch looks too extreme or
# too subtle to read as visually distinct from the standard policy.
LOW_HEIGHT_TARGET_M = 0.30


@configclass
class RewardsCfg(BaseRewardsCfg):
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-10.0,
        params={"target_height": LOW_HEIGHT_TARGET_M},
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

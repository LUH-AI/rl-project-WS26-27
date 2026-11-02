"""Go2 velocity task variant: identical to the standard config
(velocity_env_cfg.RobotEnvCfg) except for two added reward terms that
together bias the policy toward a slow, low, sequential "crawl" gait -
one leg lifts at a time, three feet always on the ground, body held lower
than standard but not collapsed. Everything else (terrain, observations,
actions, other rewards, terminations, curriculum) is inherited unchanged.

Combines the two mechanisms used separately by the Pace and LowHeight
variants (mdp.feet_gait, mdp.base_height_l2) rather than extending either
of them directly, to avoid diamond inheritance between two sibling
configclass subclasses.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg import RewardsCfg as BaseRewardsCfg
from .velocity_env_cfg import RobotEnvCfg as BaseRobotEnvCfg

# Offsets assume the contact sensor resolves ".*_foot" bodies in FR, FL, RR,
# RL order (see the same caveat in velocity_env_cfg_pace.py) - verify
# against the sensor's actual resolved body_names before trusting this.
#
# With threshold=0.75 (75% of each leg's cycle in stance, 25% swinging) and
# these offsets, each leg's 0.25-cycle swing window falls in a different
# quarter of the shared period with no overlap: RR swings first, then RL,
# then FL, then FR, each one finishing before the next starts. Exactly one
# foot is ever in the air, the other three stay planted - the standard
# definition of a crawl/creep/"singlefoot" gait, as opposed to trot/pace/
# bound's two-legs-at-once pattern.
CRAWL_OFFSETS = [0.0, 0.25, 0.75, 0.5]  # FR, FL, RR, RL
CRAWL_THRESHOLD = 0.75  # duty factor: 3 feet down at all times

# Starting point, not a measured value - slower than pace/trot's implicit
# cadence since a crawl reads as slow and deliberate. Tune if it looks too
# rushed or too dragging.
CRAWL_PERIOD_S = 1.0

# Starting weight, not validated - see the same caveat in
# velocity_env_cfg_pace.py regarding air_time_variance/feet_air_time
# fighting an under-weighted gait term.
CRAWL_GAIT_WEIGHT = 1.0

# Nominal spawn/standing height for this asset is 0.4 m (see
# velocity_env_cfg_low_height.py). This is lower than that variant's 0.30 m
# target, since "crawling" should read as flatter than just "a lower
# stance". The existing base_contact termination (robot's base body
# touching the ground ends the episode) and undesired_contacts penalty
# already in the inherited reward/termination set are what should stop
# training from just collapsing to the floor to minimize this term - watch
# the rendered rollout to confirm that holds rather than assuming it.
CRAWL_HEIGHT_TARGET_M = 0.22

# v1 at weight -10 (H1's precedent): mean height 0.308 m.
# v2 at weight -40 (4x): mean height 0.289 m - only 0.019 m of further
# movement for a 4x weight increase, a steep diminishing return. That
# pattern suggests the wall isn't "weight too low" so much as the policy
# trading off against track_lin_vel_xy/energy/joint_torques, which get
# more expensive to satisfy the lower and more bent-kneed the stance gets
# while still walking. v3 pushes weight much harder (3x again) to see
# whether there's real room left or this plateaus - check the logged
# rollout height again rather than assuming either outcome.


@configclass
class RewardsCfg(BaseRewardsCfg):
    gait = RewTerm(
        func=mdp.feet_gait,
        weight=CRAWL_GAIT_WEIGHT,
        params={
            "period": CRAWL_PERIOD_S,
            "offset": CRAWL_OFFSETS,
            "threshold": CRAWL_THRESHOLD,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
        },
    )
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-120.0,
        params={"target_height": CRAWL_HEIGHT_TARGET_M},
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

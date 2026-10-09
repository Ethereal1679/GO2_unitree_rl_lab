import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
# from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_23DOF_CFG as ROBOT_CFG
from unitree_rl_lab.assets.robots.bm_g1_23dof import G1_CYLINDER_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.terrains.terrains_cfg import COMPLEX_RANDOM_CFG
from unitree_rl_lab.tasks.locomotion.height_scan.height_scan_visualization import HEIGHT_SCAN_MARKER_CFG
from unitree_rl_lab.tasks.locomotion.mdp.commands.velocity_command import PoseVelocityCommandCfg

@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",  # "plane", "generator"
        terrain_generator=COMPLEX_RANDOM_CFG,  # None, ROUGH_TERRAINS_CFG
        max_init_terrain_level=COMPLEX_RANDOM_CFG.num_rows - 1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        # TODO
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(
            resolution=0.1,
            size=[1.5, 1.0],
            ordering="xy",
        ),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        visualizer_cfg=HEIGHT_SCAN_MARKER_CFG,
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 5.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = PoseVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        debug_vis=False,
        velocity_control_stiffness=2.0,
        heading_control_stiffness=2.0,
        rel_standing_envs=0.05, # 5%的环境站立
        ranges=PoseVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-1.0, 1.0),
        ),
        random_velocity_terrain=[],
        # Sample from the four cardinal border groups, biased toward the current heading.
        target_patch_names=(
            "target_center",
            "target_pos_x",
            "target_neg_x",
            "target_pos_y",
            "target_neg_y",
        ),
        # 按照锥形扫描选择前方的patch，尽量减少选择后方的patch
        prefer_forward_targets=False,
        target_front_cone_half_angle=math.radians(60.0),
        target_min_distance=1.0,
        target_heading_bias=4.0,
        # 随机clamp速度上限，不代表真实速度
        # NOTE 这里都是速度上限随机化，不存在负数
        velocity_ranges={
            "hf_gaps": {
                "lin_vel_x": (0.45, 1.0),
                "lin_vel_y": (0.0, 0.0),
                "ang_vel_z": (-1.0, 1.0),
            },
            "hf_steppingstones": {
                "lin_vel_x": (0.45, 1.0),
                "lin_vel_y": (0.0, 0.0),
                "ang_vel_z": (-1.0, 1.0),
            },
        },
        # 最好是正向速度，避免负向速度导致的机器人后退
        only_positive_lin_vel_x=True,
        lin_vel_threshold=0.0,
        ang_vel_threshold=0.0,
        target_dis_threshold=0.4,
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action)
        # gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.8})
        height_scanner = ObsTerm(
            func=mdp.map_scan_points,
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "asset_cfg": SceneEntityCfg("robot"),
                "grid_shape": (16, 11),
                "noise": True,
            },
            clip=(-100.0, 100.0),
        )

        def __post_init__(self):
            # self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.01, clip=(-100, 100))
        last_action = ObsTerm(func=mdp.last_action)
        height_scanner = ObsTerm(
            func=mdp.map_scan_points,
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "asset_cfg": SceneEntityCfg("robot"),
                "grid_shape": (16, 11),
                "noise": False,
            },
            clip=(-100.0, 100.0),
        )

        # def __post_init__(self):
        #     self.history_length = 5

    # privileged observations
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=2.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)})
    track_ang_vel_z = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=1.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)})
    heading_error = RewTerm(func=mdp.heading_error, weight=-2.0,) # 如果target生成在后方，机器人会学会原地转弯利用vel的正向clamp来始终追踪小速度奖励


    # -- basic
    # alive = RewTerm(func=mdp.is_alive, weight=0.1)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2e-5)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0) # protect from committing suicide
    base_height = RewTerm(func=mdp.base_height_l2, weight=-10.0, params={"target_height": 0.78})

    # -- hurry
    dont_wait = RewTerm(func=mdp.dont_wait, weight=-0.5, params={"command_name": "base_velocity"})
    stand_still = RewTerm(func=mdp.stand_still, weight=-0.5, params={"command_name": "base_velocity"})

    # -- robot
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.5)
    joint_position_penalty = RewTerm(func=mdp.joint_position_penalty, weight=-0.5, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*"),"stand_still_scale": 5.0,"velocity_threshold": 0.3,},)
    undesired_contacts = RewTerm(func=mdp.undesired_contacts,weight=-1.0,params={"threshold": 1,"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),},)
    gap_penetration = RewTerm(func=mdp.GapPenetrationPenalty, weight=-5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "min_depth": 0.05, "max_depth": 2.0,
            "depth_scale": 0.40, "duration_scale": 0.50,
            "sub_terrain_size": COMPLEX_RANDOM_CFG.size,
            "plane_z": -0.07,
            "plane_thickness": 0.02,
        },
    )

    # -- feet
    feet_air_time = RewTerm(func=mdp.feet_air_time,weight=0.1, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),"command_name": "base_velocity","threshold": 0.5,},)
    feet_slide = RewTerm(func=mdp.feet_slide, weight=-0.2,params={"asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),},)
    feet_clearance = RewTerm(func=mdp.foot_clearance_reward, weight=1.0,params={"std": 0.05,"tanh_mult": 2.0,"target_height": 0.1,"asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),},)
    feet_stumble = RewTerm(func=mdp.feet_stumble, weight=-1.0,params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),},) 

    # -- joints
    joint_deviation_arms = RewTerm(func=mdp.joint_deviation_l1, weight=-0.1,params={"asset_cfg": SceneEntityCfg("robot",joint_names=[".*_shoulder_.*_joint",".*_elbow_joint",".*_wrist_.*",],)},)
    joint_deviation_waists = RewTerm(func=mdp.joint_deviation_l1, weight=-1.0,params={"asset_cfg": SceneEntityCfg("robot",joint_names=["waist.*",],)},)
    joint_deviation_legs = RewTerm(func=mdp.joint_deviation_l1, weight=-1.0,params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},)


    # gait = RewTerm(
    #     func=mdp.feet_gait,
    #     weight=0.5,
    #     params={
    #         "period": 0.8,
    #         "offset": [0.0, 0.5],
    #         "threshold": 0.55,
    #         "command_name": "base_velocity",
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
    #     },
    # )





@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(func=mdp.illegal_contact, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso_link"), "threshold": 1.0},)
    # base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    # bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    terrain_levels = CurrTerm(func=mdp.terrain_levels_tracking_exp_vel, params={"lin_vel_threshold": (0.3, 0.6), "ang_vel_threshold": (0.1, 0.3)})
    # terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    # lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 10
        self.scene.terrain.terrain_generator.num_cols = max(5, len(self.scene.terrain.terrain_generator.sub_terrains))
        # self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.scene.terrain.max_init_terrain_level = self.scene.terrain.terrain_generator.num_rows
        self.scene.height_scanner.debug_vis = True # 开启调试可视化
        self.commands.base_velocity.debug_vis = True # 开启调试可视化

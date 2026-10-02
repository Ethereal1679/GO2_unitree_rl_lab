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
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from unitree_rl_lab.assets.robots.unitree import UNITREE_GO2_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.terrains.terrains_generator_cfg import COMPLEX_RANDOM_CFG
from unitree_rl_lab.tasks.locomotion.terrains.height_scan_visualization import HEIGHT_SCAN_MARKER_CFG
from unitree_rl_lab.tasks.locomotion.mdp.commands.velocity_command import PoseVelocityCommandCfg

@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",  # "plane", "generator"
        terrain_generator=COMPLEX_RANDOM_CFG,
        max_init_terrain_level=1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.72, 0.74, 0.77),
            roughness=0.85,
            metallic=0.0,
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    # 26 x 16 map points, represented in the robot base frame.

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)), #为了让 ray 起点位于高处，不参与最终 height scan 数值
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
            "static_friction_range": (0.3, 1.2),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.15),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
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
        interval_range_s=(5.0, 10.0),
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
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True, clip={".*": (-100.0, 100.0)}
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100), noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100), noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100), noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100), noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))
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

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100))
        velocity_commands = ObsTerm(func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100))
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.01, clip=(-100, 100))
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))
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
    heading_error = RewTerm(func=mdp.heading_error, weight=-1.0,) # 如果target生成在后方，机器人会学会原地转弯利用vel的正向clamp来始终追踪小速度奖励
    # is_alive = RewTerm(func=mdp.is_alive, weight=1.0)

    # -- base
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2e-5)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0) # protect from committing suicide

    # -- push
    dont_wait = RewTerm(func=mdp.dont_wait, weight=-0.5, params={"command_name": "base_velocity"})
    stand_still = RewTerm(func=mdp.stand_still, weight=-0.5, params={"command_name": "base_velocity"})

    # -- robot
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.5)
    joint_position_penalty = RewTerm(func=mdp.joint_position_penalty, weight=-0.5, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*"),"stand_still_scale": 5.0,"velocity_threshold": 0.3,},)
    feet_slide = RewTerm(func=mdp.feet_slide,weight=-0.1, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),},)
    feet_air_time = RewTerm(func=mdp.feet_air_time,weight=0.1, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),"command_name": "base_velocity","threshold": 0.5,},)
    air_time_variance_penalty = RewTerm(func=mdp.air_time_variance_penalty, weight=-1.0, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},)
    undesired_contacts = RewTerm(func=mdp.undesired_contacts, weight=-1.0, params={"threshold": 1, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Head_.*", ".*_hip", ".*_thigh", ".*_calf"]),},)
    feet_stumble = RewTerm(func=mdp.feet_stumble,weight=-1.0,params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),},) 
    # gap_penetration = RewTerm(func=mdp.GapPenetrationPenalty, weight=-3.0,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("height_scanner"),
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*_(foot|calf|thigh|hip)|base"),
    #         "support_radius": 0.35, # 有效半径，单位 m，表示在这个半径范围内的地面点会被认为是支撑点
    #         "min_depth": 0.05, "max_depth": 2.0,
    #         "depth_scale": 0.40, "duration_scale": 0.50,
    #     },
    # )


    # feet_contact_forces = RewTerm(
    #     func=mdp.contact_forces,
    #     weight=-0.02,
    #     params={
    #         "threshold": 100.0,
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
    #     },
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    # NOTE Termination terms don't need too many
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(func=mdp.illegal_contact, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},)
    # bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.707}) # 45 degrees
    # base_height = DoneTerm(func=mdp.root_height_below_minimum,params={"minimum_height": 0.16,"asset_cfg": SceneEntityCfg("robot"),},)


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    # terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    terrain_levels = CurrTerm(func=mdp.terrain_levels_tracking_exp_vel, params={"lin_vel_threshold": (0.3, 0.6), "ang_vel_threshold": (0.1, 0.3)})



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
        # Frame the complete 32 m modular pillar tile in the interactive viewer.
        self.viewer.eye = (38.0, 38.0, 30.0)
        self.viewer.lookat = (16.0, 16.0, 0.0)
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
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt # TODO 需要对齐真机10hz吗？

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.base_com = None


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 15
        # Keep at least one curriculum column for every configured terrain type.
        self.scene.terrain.terrain_generator.num_cols = max(5, len(self.scene.terrain.terrain_generator.sub_terrains))
        self.scene.terrain.max_init_terrain_level = self.scene.terrain.terrain_generator.num_rows // 2 # 地形一半
        self.scene.height_scanner.debug_vis = True # 开启调试可视化
        self.commands.base_velocity.debug_vis = True # 开启调试可视化

import math
from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkersCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from .reference_command import ReferenceCommand
from .compliance_augmented_reference_command import ComplianceAugmentedReferenceCommand
from .fixed_stiffness_command import FixedStiffnessCommand

@configclass
class ReferenceCommandCfg(CommandTermCfg):
    """Configuration for the joint position command generator."""

    class_type: type = ReferenceCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""
    
    @configclass
    class Ranges:
        """Uniform distribution ranges for the joint position commands."""

        min_pos: list[float] = MISSING
        """Minimum position for each joint."""
        max_pos: list[float] = MISSING
        """Maximum position for each joint."""
        
    ranges: Ranges = None
    """Distribution ranges for the joint position commands."""
    
    @configclass
    class ProcRanges:
        """Uniform distribution ranges for the procedurally generated references."""
        min_freq: float = MISSING
        max_freq: float = MISSING
        min_height: float = MISSING
        max_height: float = MISSING
        min_vel: float = MISSING
        max_vel: float = MISSING
        min_yaw_vel: float = MISSING
        max_yaw_vel: float = MISSING
        min_pitch: float = MISSING
        max_pitch: float = MISSING
        standing_prob: float = MISSING
        stance_duration: float = MISSING
        swing_height: float = MISSING
        num_steps_to_stand: int = MISSING
        dynamic_standing_only: bool = MISSING
        min_standing_pitch: float = MISSING
        max_standing_pitch: float = MISSING
        min_standing_height: float = MISSING
        max_standing_height: float = MISSING
        random_arms: bool = MISSING

    proc_ranges: ProcRanges = MISSING
    
    @configclass
    class VelocityRanges:
        """Uniform distribution ranges for the command update speed."""
            
        min_vel: list[float] = MISSING
        """Minimum velocity for each joint."""
        max_vel: list[float] = MISSING
        """Maximum velocity for each joint."""
    
    @configclass
    class JointConfig:
        """Joint configuration for the reference command."""

        num_joints: int = MISSING
        """Number of joints in the asset."""
        thigh_length: float = MISSING
        calf_length: float = MISSING   
        robot_type: str = MISSING     

        @configclass
        class LegIndicesConfig: 
            hip_yaw: int = MISSING
            hip_roll: int = MISSING
            hip_pitch: int = MISSING
            knee: int = MISSING
            ankle_pitch: int = MISSING
        
        @configclass
        class ArmIndicesConfig:
            shoulder_pitch: int = MISSING
            shoulder_roll: int = MISSING
            shoulder_yaw: int = MISSING
            elbow: int = MISSING
            wrist_pitch: int = MISSING
            wrist_roll: int = MISSING
            wrist_yaw: int = MISSING

        left_leg_indices: LegIndicesConfig = MISSING
        right_leg_indices: LegIndicesConfig = MISSING
        left_arm_indices: ArmIndicesConfig = MISSING
        right_arm_indices: ArmIndicesConfig = MISSING

    joint_config: JointConfig = MISSING

    # TODO(remove)
    n_future_steps: int = MISSING
    future_dt: float = MISSING

    demo_recording_path: str | list[str] = MISSING
    demo_playback_mode: str = MISSING
    demo_start_range: float = MISSING
    demo_stand_height: float = 0.0
    demo_pitch_angle: float = 0.0
    demo_random_reset: bool = MISSING
    demo_zero_reset: bool = False
    demo_base_policy_path: str = None
    demo_random_init: bool = False
    demo_joint_order_mode: str = "default"
    demo_upper_body_only: bool = False
    demo_zero_xy: bool = False
    demo_reset_robot_on_clip_end: bool = True
    demo_loader_class: str = "ProceduralMotionLibFromDemo"
    """Path to the demo recording file."""

    clip_length: float = None

    keypoint_body_ids: list[int] = MISSING
    goal_pose_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/goal_marker",
        markers={
            f"goal_{i}": sim_utils.SphereCfg(
                radius=0.03,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
            ) for i in range(14)
        },
    )
    local_goal_pose_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/local_goal_marker",
        markers={
            f"local_goal_{i}": sim_utils.SphereCfg(
                radius=0.03,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ) for i in range(14)
        },
    )
    local_horizon_goal_pose_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/local_horizon_goal_marker",
        markers={
            f"local_horizon_goal_{i}_horizon_{h}": sim_utils.SphereCfg(
                radius=0.03,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, h / 20.0, 0.0)),
            ) for h in range(10) for i in range(14)
        },
    )
    current_pose_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/current_marker",
        markers={
            f"kp_{i}": sim_utils.SphereCfg(
                radius=0.03,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
            ) for i in range(14)
        },
    )
    """The configuration for the goal pose visualization marker. Defaults to a DexCube marker."""

@configclass
class ComplianceAugmentedReferenceCommandCfg(ReferenceCommandCfg):
    """Configuration for the ComplianceAugmentedReferenceCommand."""
    
    class_type: type = ComplianceAugmentedReferenceCommand
    """The class type of the command generator."""

    override_reference: bool = True
    force_computation_mode: str = "feedforward" # ["feedforward", "recompute"]
    debug_vis_keypoints: bool = False
    enable_metrics: bool = True

    force_vector_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/force_vector_marker",
        markers={
            "arrow": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.3, 0.3, 0.3),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
            )
        }
    )
    adapted_keypoints_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/adapted_keypoints_marker",
        markers={
            f"kp": sim_utils.SphereCfg(
                radius=0.05,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
            )# for i in range(14)
        },
    )
    recomputed_force_vector_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/recomputed_force_marker",
        markers={
            "arrow": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.3, 0.3, 0.3),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 1.0)),
            )
        }
    )
    # torque_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
    #     prim_path="/Visuals/Command/torque_visualizer_marker",
    #     markers={
    #         f"cylinder": sim_utils.CylinderCfg(
    #             radius=0.05,
    #             height=0.2,
    #             visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
    #         )# for i in range(14)
    #     },
    # )
    ff_setpoint_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/ff_setpoint_visualizer",
        markers={
            f"ff_setpoint": sim_utils.SphereCfg(
                radius=0.05,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            )
        },
    )
    # ff_plane_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
    #     prim_path="/Visuals/Command/ff_plane_visualizer",
    #     markers={
    #         f"ff_plane": sim_utils.CuboidCfg(
    #             size=(0.1, 0.1, 0.01),
    #             visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
    #         )
    #     }
    # )
    # target_torque_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
    #     prim_path="/Visuals/Command/target_torque_marker",
    #     markers={
    #         "arrow": sim_utils.UsdFileCfg(
    #             usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
    #             scale=(0.3, 0.3, 0.3),
    #             visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.0)), # Cyan
    #         )
    #     }
    # )
    # computed_torque_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
    #     prim_path="/Visuals/Command/computed_torque_marker",
    #     markers={
    #         "arrow": sim_utils.UsdFileCfg(
    #             usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
    #             scale=(0.3, 0.3, 0.3),
    #             visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.75, 0.4)), # Magenta
    #         )
    #     }
    # )


@configclass
class FixedStiffnessCommandCfg(CommandTermCfg):
    """Configuration for the ObstacleMotionCommand."""

    class_type: type = FixedStiffnessCommand

    desired_stiffness: float = MISSING
    desired_rotational_stiffness: float = MISSING
    forcefield_stiffness: float = MISSING
    forcefield_rotational_stiffness: float = MISSING

"""Registration of deployable SoftMimic tasks."""

import gymnasium as gym

from . import agents, g1_force_control

_ENV_PREFIX = "Isaac-G1-AugmentedReference-ForceTorque-Control-VariableStiffness"
REGISTERED_TASK_IDS = []


def _compose_task_id(motion_meta: dict, mode_meta: dict) -> str:
    parts = [_ENV_PREFIX]
    motion_segment = motion_meta.get("id_segment")
    if motion_segment:
        parts.append(motion_segment)
    parts.append(mode_meta["id_segment"])
    return "-".join(parts) + "-Deployable-Mimic-v0"


for (motion_key, mode_key), env_cfg_cls in g1_force_control.AVAILABLE_DEPLOYABLE_ENV_CFGS.items():
    motion_meta = g1_force_control.SUPPORTED_MOTION_VARIANTS[motion_key]
    mode_meta = g1_force_control.SUPPORTED_MODE_VARIANTS[mode_key]
    task_id = _compose_task_id(motion_meta, mode_meta)
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg_cls,
            "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.G1StuntRunnerCfg,
        },
    )
    REGISTERED_TASK_IDS.append(task_id)

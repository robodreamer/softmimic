# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

# from softmimic_gym.rsl_rl_extensions.rl_cfg import RslRlPpoStudentTeacherCfg, RslRlDistillAlgorithmCfg, RslRlPpoActorCriticPULSEHierarchicalCfg, RslRlPpoStudentTeacherPULSECfg, RslRlSymmetryCfg, RslRlPpoSymmetryAugAlgorithmCfg, RslRlPpoAMPAlgorithmCfg
import pathlib

@configclass
class G1WBCRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 100000
    save_interval = 500
    experiment_name = "g1_rough"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 512, 256, 128],
        critic_hidden_dims=[512, 512, 512, 512],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002, #0.003,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

@configclass
class G1StuntRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 100000
    save_interval = 500
    experiment_name = "g1_rough"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 512, 256, 128],
        critic_hidden_dims=[512, 512, 512, 512],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002, #0.003,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        # learning_rate=3.0e-4,
        schedule="adaptive",
        # schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

@configclass
class G1CSEStuntRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 100000
    save_interval = 500
    experiment_name = "g1_rough"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 512, 256, 128],
        critic_hidden_dims=[512, 512, 512, 512],
        activation="elu",
        class_name="ActorCriticCSE",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002, #0.003,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        class_name="PPOCSE",
    )

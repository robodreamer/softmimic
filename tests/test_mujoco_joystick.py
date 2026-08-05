import glfw
import numpy as np
import torch

from softmimic_deploy.src.interfaces.mujoco_interface import MujocoInterface
from softmimic_deploy.src.joysticks.mujoco_joystick import MujocoJoystick
from softmimic_deploy.src.sensors.desired_rotational_stiffness_log_sensor import (
    DesiredRotationalStiffnessLogSensor,
)
from softmimic_deploy.src.sensors.desired_stiffness_log_sensor import (
    DesiredStiffnessLogSensor,
)
from softmimic_deploy.src.sensors.future_reference_ang_vel_sensor import (
    FutureReferenceAngVelSensor,
)
from softmimic_deploy.src.sensors.future_reference_xy_vel_sensor import (
    FutureReferenceXYVelSensor,
)
from softmimic_deploy.src.sensors.reference_ang_vel_sensor import ReferenceAngVelSensor
from softmimic_deploy.src.sensors.reference_xy_vel_sensor import ReferenceXYVelSensor


def test_lifecycle_keys_do_not_use_mujoco_viewer_shortcuts():
    joystick = MujocoJoystick()

    joystick.key_callback(glfw.KEY_F8)
    assert joystick.get_buttons()[1] == 1

    joystick.key_callback(glfw.KEY_F9)
    np.testing.assert_array_equal(joystick.get_abxy(), [0, 0, 1, 0])

    joystick.key_callback(glfw.KEY_F10)
    np.testing.assert_array_equal(joystick.get_abxy(), [0, 0, 0, 1])

    joystick.key_callback(glfw.KEY_F11)
    np.testing.assert_array_equal(joystick.get_abxy(), [1, 0, 0, 0])


def test_number_row_and_numpad_motion_keys_and_reset():
    joystick = MujocoJoystick()

    joystick.key_callback(glfw.KEY_8)
    joystick.key_callback(glfw.KEY_KP_4)
    joystick.key_callback(glfw.KEY_3)
    np.testing.assert_allclose(joystick.get_command(), [0.2, 0.0, -0.2, 0.1])

    joystick.key_callback(glfw.KEY_5)
    np.testing.assert_allclose(joystick.get_command(), [0.0, 0.0, 0.0, 0.0])


def test_motion_commands_are_bounded():
    joystick = MujocoJoystick()

    for _ in range(20):
        joystick.key_callback(glfw.KEY_8)
        joystick.key_callback(glfw.KEY_6)

    np.testing.assert_allclose(joystick.get_command(), [0.8, 0.0, 1.0, 0.0])


def test_stiffness_keys_update_and_reset_policy_observations():
    joystick = MujocoJoystick()
    interface = MujocoInterface.__new__(MujocoInterface)
    interface.joystick = joystick

    joystick.key_callback(glfw.KEY_9)
    assert joystick.get_stiffness() == 75.0
    np.testing.assert_allclose(
        DesiredStiffnessLogSensor(interface).get_data(),
        np.log([75.0]),
    )
    np.testing.assert_allclose(
        DesiredRotationalStiffnessLogSensor(interface).get_data(),
        np.log([1.25]),
    )

    joystick.key_callback(glfw.KEY_0)
    assert joystick.get_stiffness() == joystick.DEFAULT_STIFFNESS


class _ReferenceInterface:
    def get_reference_velocity_offset(self):
        return np.array([0.2, 0.0, -0.4], dtype=np.float32)


class _ReferenceMotion:
    root_vel = torch.tensor([1.0, 2.0, 0.0])
    root_ang_vel = torch.tensor([0.0, 0.0, 0.5])
    future_root_vel = torch.zeros((20, 3))
    future_root_ang_vel = torch.zeros((20, 3))


def test_reference_velocity_override_preserves_policy_dimensions():
    interface = _ReferenceInterface()
    motion = _ReferenceMotion()

    np.testing.assert_allclose(
        ReferenceXYVelSensor(interface, wholeexo_sensor=motion).get_data(),
        [1.2, 2.0],
    )
    np.testing.assert_allclose(
        ReferenceAngVelSensor(interface, wholeexo_sensor=motion).get_data(),
        [0.1],
    )

    future_xy = FutureReferenceXYVelSensor(interface, wholeexo_sensor=motion).get_data()
    future_yaw = FutureReferenceAngVelSensor(interface, wholeexo_sensor=motion).get_data()
    assert future_xy.shape == (20, 2)
    assert future_yaw.shape == (20, 1)
    np.testing.assert_allclose(future_xy, np.tile([0.2, 0.0], (20, 1)))
    np.testing.assert_allclose(future_yaw, np.full((20, 1), -0.4))

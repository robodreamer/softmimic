import glfw
import numpy as np

from softmimic_deploy.src.joysticks.mujoco_joystick import MujocoJoystick


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


def test_numpad_motion_keys_and_reset():
    joystick = MujocoJoystick()

    joystick.key_callback(glfw.KEY_KP_8)
    joystick.key_callback(glfw.KEY_KP_4)
    joystick.key_callback(glfw.KEY_KP_9)
    np.testing.assert_allclose(joystick.get_command(), [0.2, 0.0, -0.2, 0.1])

    joystick.key_callback(glfw.KEY_KP_5)
    np.testing.assert_allclose(joystick.get_command(), [0.0, 0.0, 0.0, 0.0])

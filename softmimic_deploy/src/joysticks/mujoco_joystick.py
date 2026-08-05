import glfw
import numpy as np


class MujocoJoystick:
    KEYMAP = """
SoftMimic controls (focus the MuJoCo viewer)
  F8               Initialize / calibrate pose
  F9               Start / resume policy
  F10              Stop policy / damping mode
  F11              Recalibrate while running

  8 / 2            Increase / decrease forward velocity offset
  4 / 6            Increase left / right yaw-rate offset
  3 / 1            Increase / decrease height command
  9 / 7            Increase / decrease desired policy stiffness
  5                Zero velocity and height commands
  0                Reset stiffness to its default

  The top number row and numeric keypad are both accepted.

  Note: Steering offsets the recorded reference velocities. It is experimental
        and is most useful with the bundled walk policy.

Mouse perturbations (while the policy is running)
  Double left-click            Select a robot body
  Ctrl + right-drag            Apply force in the vertical plane
  Ctrl + Shift + right-drag    Apply force in the horizontal plane
  Ctrl + left-drag             Apply torque
""".strip()

    COMMAND_STEP = 0.2
    HEIGHT_STEP = 0.1
    MIN_HEIGHT_OFFSET = -0.35
    MAX_HEIGHT_OFFSET = 0.15
    MIN_LINEAR_COMMAND = -0.8
    MAX_LINEAR_COMMAND = 0.8
    MIN_YAW_COMMAND = -1.0
    MAX_YAW_COMMAND = 1.0
    DEFAULT_STIFFNESS = 60.0
    MIN_STIFFNESS = 40.0
    MAX_STIFFNESS = 800.0
    STIFFNESS_FACTOR = 1.25
    
    def __init__(self):
        self.mode = 0
        self.ctrlmode_left = 0
        self.ctrlmode_right = 0
        self.left_stick = [0, 0]
        self.right_stick = [0, 0]
        self.left_upper_switch = 0
        self.left_lower_left_switch = 0
        self.left_lower_right_switch = 0
        self.right_upper_switch = 0
        self.right_lower_left_switch = 0
        self.right_lower_right_switch = 0
        self.left_upper_switch_pressed = 0
        self.left_lower_left_switch_pressed = 0
        self.left_lower_right_switch_pressed = 0
        self.right_upper_switch_pressed = 0
        self.right_lower_left_switch_pressed = 0
        self.right_lower_right_switch_pressed = 0
        self.current_policy = 1
        self.stiffness = self.DEFAULT_STIFFNESS
        self.running = False
        self.run_thread = None
        self.root = None
        self.step_counter = 0
        
        self.a_button, self.b_button, self.x_button, self.y_button = 0, 0, 0, 0

        # smoothing
        self.left_stick_smooth = [0, 0]
        self.right_stick_smooth = [0, 0]
    
    def key_callback(self, key):
        """Translate non-conflicting viewer keys into joystick inputs."""
        command_changed = True
        if key in (glfw.KEY_5, glfw.KEY_KP_5):
            self.left_stick[0] = 0
            self.left_stick[1] = 0
            self.right_stick[0] = 0
            self.right_stick[1] = 0
        elif key in (glfw.KEY_8, glfw.KEY_KP_8):
            self.left_stick[1] = min(
                self.left_stick[1] + self.COMMAND_STEP,
                self.MAX_LINEAR_COMMAND,
            )
        elif key in (glfw.KEY_2, glfw.KEY_KP_2):
            self.left_stick[1] = max(
                self.left_stick[1] - self.COMMAND_STEP,
                self.MIN_LINEAR_COMMAND,
            )
        elif key in (glfw.KEY_4, glfw.KEY_KP_4):
            self.right_stick[0] = max(
                self.right_stick[0] - self.COMMAND_STEP,
                self.MIN_YAW_COMMAND,
            )
        elif key in (glfw.KEY_6, glfw.KEY_KP_6):
            self.right_stick[0] = min(
                self.right_stick[0] + self.COMMAND_STEP,
                self.MAX_YAW_COMMAND,
            )
        elif key in (glfw.KEY_3, glfw.KEY_KP_3):
            self.right_stick[1] = min(
                self.right_stick[1] + self.HEIGHT_STEP,
                self.MAX_HEIGHT_OFFSET,
            )
        elif key in (glfw.KEY_1, glfw.KEY_KP_1):
            self.right_stick[1] = max(
                self.right_stick[1] - self.HEIGHT_STEP,
                self.MIN_HEIGHT_OFFSET,
            )
        elif key in (glfw.KEY_9, glfw.KEY_KP_9):
            self.stiffness = min(
                self.stiffness * self.STIFFNESS_FACTOR,
                self.MAX_STIFFNESS,
            )
        elif key in (glfw.KEY_7, glfw.KEY_KP_7):
            self.stiffness = max(
                self.stiffness / self.STIFFNESS_FACTOR,
                self.MIN_STIFFNESS,
            )
        elif key in (glfw.KEY_0, glfw.KEY_KP_0):
            self.stiffness = self.DEFAULT_STIFFNESS
        elif key == glfw.KEY_F11:
            self.a_button = 1
            command_changed = False
        elif key == glfw.KEY_F9:
            self.x_button = 1
            command_changed = False
        elif key == glfw.KEY_F10:
            self.y_button = 1
            command_changed = False
        elif key == glfw.KEY_F8:
            self.left_upper_switch = 1
            command_changed = False
        else:
            command_changed = False

        if command_changed:
            print(
                "[SoftMimic] "
                f"forward_offset={self.left_stick[1]:+.2f} m/s, "
                f"yaw_offset={self.right_stick[0]:+.2f} rad/s, "
                f"height={self.right_stick[1] + 0.75:.2f} m, "
                f"stiffness={self.stiffness:.1f}"
            )
    
    def update_stick(self, stick, x, y):
        if stick == 'left':
            self.left_stick = [x, y]
        elif stick == 'right':
            self.right_stick = [x, y]
    
    def update_switch(self, switch, state):
        if switch == 'left_upper':
            self.left_upper_switch = state
        elif switch == 'right_upper':
            self.right_upper_switch = state
        elif switch == 'left_lower':
            self.left_lower_left_switch = state
            self.left_lower_left_switch_pressed = state
        elif switch == 'right_lower':
            self.right_lower_right_switch = state
            self.right_lower_right_switch_pressed = state
    
    def get_buttons(self):
        buttons = np.array([self.left_lower_left_switch, self.left_upper_switch, self.right_lower_right_switch, self.right_upper_switch])
        self.left_lower_left_switch, self.left_upper_switch, self.right_lower_right_switch, self.right_upper_switch = 0, 0, 0, 0
        return buttons


    def get_abxy(self):
        abxy = np.array([self.a_button, self.b_button, self.x_button, self.y_button])
        self.a_button, self.b_button, self.x_button, self.y_button = 0, 0, 0, 0
        return abxy
    
    def get_command(self):
        # self.left_stick_smooth = [0.8 * self.left_stick_smooth[0] + 0.2 * self.left_stick[0], 0.8 * self.left_stick_smooth[1] + 0.2 * self.left_stick[1]]
        # self.right_stick_smooth = [0.8 * self.right_stick_smooth[0] + 0.2 * self.right_stick[0], 0.8 * self.right_stick_smooth[1] + 0.2 * self.right_stick[1]]

        # return [self.left_stick_smooth[1], self.left_stick_smooth[0], self.right_stick_smooth[0], self.right_stick_smooth[1]]

        return [self.left_stick[1], self.left_stick[0], self.right_stick[0], self.right_stick[1]]

    def get_current_policy(self):
        return self.current_policy

    def get_stiffness(self):
        return self.stiffness

    def get_rotational_stiffness(self):
        return self.stiffness / self.DEFAULT_STIFFNESS

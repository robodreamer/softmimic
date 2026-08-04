import glfw
import numpy as np


class MujocoJoystick:
    KEYMAP = """
SoftMimic controls (focus the MuJoCo viewer)
  F8               Initialize / calibrate pose
  F9               Start / resume policy
  F10              Stop policy / damping mode
  F11              Recalibrate while running

  Numpad 8 / 2     Increase / decrease forward command
  Numpad 4 / 6     Increase left / right turn command
  Numpad 9 / 3     Increase / decrease height command
  Numpad 5         Zero all motion commands

Mouse perturbations (while the policy is running)
  Double left-click            Select a robot body
  Ctrl + right-drag            Apply force in the vertical plane
  Ctrl + Shift + right-drag    Apply force in the horizontal plane
  Ctrl + left-drag             Apply torque
""".strip()

    COMMAND_STEP = 0.2
    HEIGHT_STEP = 0.1
    
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
        if key == glfw.KEY_KP_5:
            self.left_stick[0] = 0
            self.left_stick[1] = 0
            self.right_stick[0] = 0
            self.right_stick[1] = 0
        elif key == glfw.KEY_KP_8:
            self.left_stick[1] += self.COMMAND_STEP
        elif key == glfw.KEY_KP_2:
            self.left_stick[1] -= self.COMMAND_STEP
        elif key == glfw.KEY_KP_4:
            self.right_stick[0] -= self.COMMAND_STEP
        elif key == glfw.KEY_KP_6:
            self.right_stick[0] += self.COMMAND_STEP
        elif key == glfw.KEY_KP_9:
            self.right_stick[1] += self.HEIGHT_STEP
        elif key == glfw.KEY_KP_3:
            self.right_stick[1] -= self.HEIGHT_STEP
        elif key == glfw.KEY_F11:
            self.a_button = 1
        elif key == glfw.KEY_F9:
            self.x_button = 1
        elif key == glfw.KEY_F10:
            self.y_button = 1
        elif key == glfw.KEY_F8:
            self.left_upper_switch = 1
    
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

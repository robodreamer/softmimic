import numpy as np

from softmimic_deploy.src.sensors.base_sensor import BaseSensor


class DesiredStiffnessLogSensor(BaseSensor):

    dim = 1

    def __init__(self, interface, scale=1.0):
        super().__init__(interface, scale=scale)

        # HARDCODED FOR NOW
        self.scale = 1.

    def get_data(self):
        if type(self.interface).__module__ == "softmimic_deploy.src.interfaces.mujoco_interface":
            stiffness = self.interface.get_stiffness_commands()
        else:
            stiffness = 60.0
        return np.log(np.array([stiffness])) * self.scale

from softmimic_deploy.src.sensors.base_sensor import BaseSensor


class FutureReferenceAngVelSensor(BaseSensor):

    dim = 1 * 20

    def __init__(self, interface, scale=1.0, wholeexo_sensor=None):
        super().__init__(interface, scale)

        assert wholeexo_sensor is not None, "wholeexo_sensor must be provided"
        self.wholeexo_sensor = wholeexo_sensor

    def get_data(self):
        reference_root_ang_vel = self.wholeexo_sensor.future_root_ang_vel[:, 2:3]
        if hasattr(self.interface, "get_reference_velocity_offset"):
            yaw_offset = reference_root_ang_vel.new_tensor(
                self.interface.get_reference_velocity_offset()[2:3]
            )
            reference_root_ang_vel = reference_root_ang_vel + yaw_offset
        return reference_root_ang_vel * self.scale

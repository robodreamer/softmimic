from softmimic_deploy.src.sensors.base_sensor import BaseSensor


class ReferenceXYVelSensor(BaseSensor):

    dim = 2

    def __init__(self, interface, scale=1.0, wholeexo_sensor=None):
        super().__init__(interface, scale)

        assert wholeexo_sensor is not None, "wholeexo_sensor must be provided"
        self.wholeexo_sensor = wholeexo_sensor

    def get_data(self):
        reference_root_xy_vel = self.wholeexo_sensor.root_vel[0:2]
        if hasattr(self.interface, "get_reference_velocity_offset"):
            velocity_offset = reference_root_xy_vel.new_tensor(
                self.interface.get_reference_velocity_offset()[0:2]
            )
            reference_root_xy_vel = reference_root_xy_vel + velocity_offset
        return reference_root_xy_vel * self.scale

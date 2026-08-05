# MuJoCo Quickstart

Run these commands from the repository checkout. The helper script resolves
the repository path itself, so the installed shortcuts also work from other
directories.

## One-command presets

Static standing policy:

```bash
./scripts/run_mujoco.sh stand
```

Recorded walking policy:

```bash
./scripts/run_mujoco.sh walk
```

Additional deployment arguments can follow the preset. For example:

```bash
./scripts/run_mujoco.sh walk --time-limit 20
./scripts/run_mujoco.sh stand --fix-base
```

Use `./scripts/run_mujoco.sh --help` to list the presets and their exact model
and motion files.

## Quick shell aliases

Load the aliases in the current Bash session:

```bash
source scripts/aliases.sh
```

Run `source` directly in Bash—do not use `uv run source scripts/aliases.sh`.
`source` is a shell built-in rather than an executable, and the preset runners
already invoke uv internally.

Then run either preset from any directory:

```bash
sm-stand
sm-walk
```

The aliases forward extra arguments:

```bash
sm-walk --time-limit 20
```

To load them automatically in new Bash sessions, add this line to
`~/.bashrc`, replacing the path with your checkout if necessary:

```bash
source "$HOME/Projects/repos/softmimic/scripts/aliases.sh"
```

## Startup sequence

1. Wait for the MuJoCo viewer and console keymap to appear.
2. Focus the viewer and press `F8` to initialize/calibrate the robot.
3. Press `F9` to deploy the policy.
4. For the recorded walk preset, press `F9` again if the reference remains at
   its standing frame.
5. Press `F10` to stop and enter damping mode, or `F11` to recalibrate.

## Policy limitations

The `stand` preset uses `StaticStand-SoftMimic` with `stand.csv`. It is intended
to hold a standing pose and respond to physical perturbations; it does not
produce walking steps.

The `walk` preset uses `GMTWalkStand-SoftMimic` with `walk.csv`. It follows the
recorded joint poses and step timing. Number keys apply experimental forward
velocity and yaw-rate offsets to the current and future reference observations.
This preserves the exported policy's input dimensions, but it is not equivalent
to a policy trained for joystick locomotion and large offsets may be unstable.

The top number row and numeric keypad use the same bindings:

- `8`/`2`: increase/decrease forward velocity offset,
- `4`/`6`: increase left/right yaw-rate offset,
- `3`/`1`: increase/decrease height,
- `9`/`7`: increase/decrease desired policy stiffness,
- `5`: zero velocity and height commands,
- `0`: reset desired stiffness to `60`.

The console prints the resulting values after each command. Desired stiffness
is provided to the policy as its existing linear and rotational stiffness
observations; it does not directly change the fixed low-level MuJoCo PD gains.

## Applying forces with the mouse

Mouse perturbations work while the policy loop is running:

- Double left-click a robot body to select it.
- Hold `Ctrl` and right-drag to apply force in the vertical plane.
- Hold `Ctrl+Shift` and right-drag to apply force in the horizontal plane.
- Hold `Ctrl` and left-drag to apply torque.

Do not pass `--fix-base` when testing whole-body recovery from pushes.

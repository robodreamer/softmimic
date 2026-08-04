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

The `walk` preset uses `GMTWalkStand-SoftMimic` with `walk.csv`. It replays the
recorded walking reference, including its step timing and direction. This
policy also has `velocity_commands: null`, so numpad commands do not steer it.

Numpad movement bindings are retained for compatible policies exported with
active velocity-command observations. Use the numeric keypad with Num Lock
enabled; the top-row number keys are not mapped.

## Applying forces with the mouse

Mouse perturbations work while the policy loop is running:

- Double left-click a robot body to select it.
- Hold `Ctrl` and right-drag to apply force in the vertical plane.
- Hold `Ctrl+Shift` and right-drag to apply force in the horizontal plane.
- Hold `Ctrl` and left-drag to apply torque.

Do not pass `--fix-base` when testing whole-body recovery from pushes.

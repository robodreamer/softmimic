# SoftMimic Code Release

This repository provides an implementation of the paper:


<td style="padding:20px;width:75%;vertical-align:middle">
      <a href="https://gmargo11.github.io/softmimic/" target="_blank">
      <b> SoftMimic: Learning Compliant Whole-body Control from Examples </b>
      </a>
      <br>
      <a href="https://gmargo11.github.io/" target="_blank">Gabriel B. Margolis</a>*, <a href="https://www.linkedin.com/in/mwangjoy/" target="_blank">Michelle Wang</a>*, <a href="https://nolie-rolie.github.io/" target="_blank">Nolan Fey</a>, and <a href="https://people.csail.mit.edu/pulkitag" target="_blank">Pulkit Agrawal</a>
      <br>
      <em>preprint</em>, 2025
      <br>
      <a href="https://arxiv.org/abs/2510.17792" target="_blank">paper</a> /
      <a href="https://gmargo11.github.io/softmimic/" target="_blank">project page</a>
    <br>
</td>

<br>

If you use this repository in your work, consider citing:

```
@article{margolis2025softmimic,
    title={{SoftMimic}: Learning Compliant Whole-body Control from Examples},
    author={Margolis, Gabriel B. and Wang, Michelle and Fey, Nolan and Agrawal, Pulkit},
    journal={arXiv preprint arXiv:2510.17792},
    year={2025}
}
```

The project is distributed under the MIT License (see `LICENSE`). Third-party components such as `unitree_sdk2` retain their original licenses in `third-party/unitree_sdk2`.

Code in `unitree_sdk2` for LCM interface comes from OpenHomie: https://github.com/InternRobotics/OpenHomie which is based on walk-these-ways: https://github.com/Improbable-AI/walk-these-ways

This code depends on a number of open-source projects without which the system would not be possible:

- IsaacLab: https://github.com/isaac-sim/IsaacLab
- RSL-RL: https://github.com/leggedrobotics/rsl_rl
- Mink: https://github.com/kevinzakka/mink
- MuJoCo: https://github.com/google-deepmind/mujoco
- unitree_sdk2: https://github.com/unitreerobotics/unitree_sdk2

### Changes in this fork

This fork keeps the original SoftMimic research code and adds deployment
quality-of-life improvements:

- a Python 3.12 pin and repository-local uv environment workflow,
- complete MuJoCo rendering dependencies, including ImageIO and FFmpeg,
- removal of an interactive debug pause during deployment startup,
- MuJoCo-safe keyboard bindings with an instruction keymap in the console,
- mouse-perturbation instructions for testing policy compliance,
- one-command standing and recorded-walking presets,
- sourceable `sm-stand` and `sm-walk` shell aliases,
- focused tests for the MuJoCo keyboard mapping.

## 1. Installation <a name="installation"></a>

Install [uv](https://docs.astral.sh/uv/) first, then clone this fork:

```bash
git clone git@github.com:robodreamer/softmimic.git
cd softmimic

uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[all]"
```

The checked-in `.python-version` keeps `uv run` on Python 3.12. The `all` extra
installs MuJoCo, ImageIO/FFmpeg video support, LCM, Mink, evaluation tools, and
development tools. You do not need to activate the environment when commands
are launched through `uv run` or the preset scripts.

To keep the original repository available for updates:

```bash
git remote add upstream git@github.com:Improbable-AI/softmimic.git
git remote -v
```

The intended remote convention is:

- `origin`: `git@github.com:robodreamer/softmimic.git`
- `upstream`: `git@github.com:Improbable-AI/softmimic.git`

Use `uv pip install --python .venv/bin/python -e "softmimic_gym[isaac]"` if
you intend to run Isaac-based training scripts.

---

## 2. Repository Layout <a name="layout"></a>

```
softmimic/
├── compliant_motion_augmentation/   # Compliant motion augmentation pipeline
├── datasets/motions_csv/             # Deployment motion references
├── pretrained_models/               # Exported policies (JIT + params)
├── scripts/                         # Quick-run, deployment, and RSL-RL helpers
├── softmimic_deploy/                # Deployment interfaces, sensors, utils
├── softmimic_gym/                   # Isaac Lab extension + packaging
├── QUICKSTART.md                    # MuJoCo presets and interaction guide
└── third-party/unitree_sdk2/        # Unitree SDK source (for deployment)
```

---

## 3. Deployment Interfaces <a name="deployment"></a>

### 3.1 MuJoCo simulation

For preset commands, shell shortcuts, the startup sequence, and mouse-force
controls, see [MuJoCo Quickstart](QUICKSTART.md).

The shortest way to launch the bundled policies is:

```bash
./scripts/run_mujoco.sh stand
./scripts/run_mujoco.sh walk
```

For optional commands that work from any directory:

```bash
source scripts/aliases.sh
sm-stand
sm-walk
```

| Preset | Exported policy | Motion | Keyboard steering |
| --- | --- | --- | --- |
| `stand` | StaticStand-SoftMimic | `stand.csv` | No |
| `walk` | GMTWalkStand-SoftMimic | `walk.csv` | No; follows recorded direction and steps |

The equivalent manual standing command is:

```bash
uv run python softmimic_deploy/src/deploy_policy_interface.py \
  --interface mujoco \
  --policy ../../pretrained_models/2025-09-26_03-54-58_StaticStand-SoftMimic/model_48000.jit \
  --motion_path stand.csv \
  --render
```

Controls (`mujoco` viewer):

- `F8`: initialize/calibrate the pose,
- `F9`: start/resume the policy,
- `F10`: stop the policy and enter damping mode,
- `F11`: recalibrate while the policy is running,
- Numpad `8`/`2`: increase/decrease the forward command (only for policies
  exported with velocity-command observations),
- Numpad `4`/`6`: increase the left/right turn command (only for policies
  exported with velocity-command observations),
- Numpad `9`/`3`: increase/decrease the height command when supported,
- Numpad `5`: zero all command values.

The included `StaticStand` and `GMTWalkStand` policies have
`velocity_commands: null`; they cannot be steered with the numpad. The walk
preset follows the position, direction, and steps recorded in `walk.csv`.

Mouse perturbations while the policy is running:

- Double left-click a robot body to select it,
- `Ctrl` + right-drag to apply force in the vertical plane,
- `Ctrl` + `Shift` + right-drag to apply force in the horizontal plane,
- `Ctrl` + left-drag to apply torque.

The same keymap is printed in the console when the MuJoCo viewer opens. These
bindings avoid MuJoCo's built-in viewer shortcuts.

The CLI accepts any motion CSV under `datasets/motions_csv/` or `SOFTMIMIC_DATA_ROOT`. Paths outside that tree can be passed explicitly.

### 3.2 Unitree G1 hardware

1. **Network setup**
   - Connect via Ethernet and assign a static IP on `192.168.123.0/24`.
   - Ensure `ssh unitree@ROBOT_IP` is reachable (default G1 `ROBOT_IP` is `192.168.123.164` and default password `123`).

2. **Repository sync**

   ```bash
   # Optional overrides: ROBOT_HOST, TARGET_DIR
   ./send_to_g1.sh
   ```

3. **On-robot environment**

   ```bash
   ssh unitree@192.168.123.164
   cd ~/softmimic_release
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
   bash Miniconda3-latest-Linux-aarch64.sh
   conda create -n softmimic python=3.10
   conda activate softmimic
   pip install -e ".[lcm]"
   sudo apt-get install liblcm-dev
   ```

4. **Unitree SDK build**

   ```bash
   cd third-party/unitree_sdk2
   mkdir -p build && cd build
   cmake ..
   make -j
   ```

5. **Deployment**

   - Terminal 1 (`./g1_control eth0`) to bridge between LCM and the Unitree SDK.
   - Terminal 2 to launch the policy:

     ```bash
     python softmimic_deploy/src/deploy_policy_interface.py \
       --interface lcm \
       --policy ../../pretrained_models/2025-09-26_03-54-58_StaticStand-SoftMimic/model_48000.jit \
       --motion_path stand.csv
     ```

   - Button guide matches the physical remote (L1: initial pose, X: deploy, Y: e-stop/damping mode).

---

## 4. Training, Evaluation, and Augmentation <a name="training"></a>

### 4.1 Compliant Motion Augmentation

Visualize the augmentation process in Mujoco:
```bash
python mink_generator_ff.py interactive     --motion_path ../datasets/motions_csv/stand.csv     --force_mode forcefield
```
```bash
python mink_generator_ff.py interactive     --motion_path ../datasets/motions_csv/boxpick.csv     --force_mode collision-emulator
```

Generate the full augmented dataset:

```bash
cd compliant_motion_augmentation
./generate_all.sh  # batch-generate datasets listed in config.py
```

To generate a compliant dataset for a new csv file, simply edit `generate_all.sh`. 

We follow the CSV formatting convention of Unitree's LAFAN1 Retargeting Dataset, appending two columns that annotate the foot contact state.


### 4.2 Isaac Lab tasks (softmimic_gym)

Install the extension then register environments:

```bash
cd softmimic_gym
pip install -e ".[isaac]"
cd ..
python scripts/list_envs.py  # prints registered task IDs
```

Training example (requires Isaac Sim running or headless):

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-G1-AugmentedReference-ForceTorque-Control-VariableStiffness-Hybrid-Deployable-Mimic-v0 --headless
```

### 4.3 Policy evaluation and video export

```bash
python scripts/rsl_rl/test.py \
  --task Isaac-G1-AugmentedReference-ForceTorque-Control-VariableStiffness-Hybrid-Deployable-Mimic-v0 \
  --num_envs 1 \
  --video --video_length 2000 --headless
```

---

## 5. Configuration and Safety Notes <a name="notes"></a>

- Follow all safety guidelines given by Unitree.
- Ensure that only one instance of the deployment script runs on the robot at a time.
- This is research code provided as-is to facilitate future research -- the user assumes all liability.

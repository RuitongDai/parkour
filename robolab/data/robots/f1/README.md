# F1 robot asset

`f1-Parkour` imports the supplied `f1_1.urdf` directly through Isaac Lab's URDF
converter. Its 28 revolute joint names match the F1 action and actuator
configuration. The Parkour terrain comes from RoboLab, so the XML scene/model
files are retained only as MuJoCo references and are not spawned by the task.
The two geometry-less `mid360`/`imu_in_pelvis` helper links were omitted because
Isaac Sim 5.1 generates unresolved visual references for empty URDF links; the
training environment mounts its depth camera directly on `torso_link`.

Visual geometry still uses the supplied STL files. For stable and fast PhysX
training, the pelvis and ankle-roll collision meshes use boxes fitted to their
STL bounds; the supplied thigh, shin, torso, and arm collisions were already
simplified boxes.

The Isaac Lab articulation and actuator configuration is defined in
`robolab/assets/robots/f1.py`. Its seven actuator groups follow the values in
the supplied `x3_constants.py`; despite that source filename, those constants
are associated with this F1 model.

## Parkour sim2sim

The F1 MuJoCo deployment uses:

- `f1_sim2sim.xml`: the actuated 28-DoF robot shared by all scenes;
- `f1_plane.xml`: flat-ground validation;
- `f1_stairs.xml`: a deterministic 10 cm up/platform/down staircase;
- `scripts/mujoco/sim2sim_f1_parkour.py`: F1 joint mapping, action scaling,
  PD control, torque limits, pelvis IMU observations, depth inference, and
  model/ONNX validation.

Export a trained F1 checkpoint as separate depth-encoder and actor graphs:

```bash
python robolab/scripts/rsl_rl/play_parkour.py \
  --task f1-Parkour-Play \
  --num_envs 1 \
  --checkpoint /absolute/path/to/model_<N>.pt \
  --exportonnx \
  --headless
```

Run the newest complete `logs/rsl_rl/f1_parkour/*/exported` pair automatically:

```bash
python robolab/scripts/mujoco/sim2sim_f1_parkour.py --scene stairs
```

The default ONNX provider is `auto`, which prefers CUDA when
`CUDAExecutionProvider` is installed and otherwise falls back to CPU. Use
`--onnx-provider cpu` only when CPU inference is explicitly wanted. To check
the MJCF and joint/actuator mapping before an ONNX export exists:

```bash
python robolab/scripts/mujoco/sim2sim_f1_parkour.py --validate-only
```

# F1 AMP motion data

Put converted F1 AMP motion files (`*.pkl`) in this directory. The
`f1-Parkour` configuration discovers every pickle automatically and gives each
file equal sampling weight.

Until at least one valid pickle exists, the task configuration can be loaded,
but environment creation/training will stop with the motion manager's
`No motion data files` error.

Each file must use the RoboLab AMP pickle schema:

- `fps`: scalar sample rate;
- `root_pos`: `(frames, 3)` world position;
- `root_rot`: `(frames, 4)` quaternion in `(w, x, y, z)` order;
- `dof_pos`: `(frames, 28)` in the F1 Isaac Sim articulation order;
- `key_body_pos`: `(frames, 6, 3)`, ordered as left/right ankle roll,
  left/right knee, left/right elbow;
- `loop_mode`: scalar loop-mode value.

## CSV conversion

Convert the headerless F1 GMR CSV files with:

```bash
conda run --no-capture-output -n robo python \
  robolab/scripts/tools/retarget/f1_csv_to_lab.py \
  --input_dir robolab/data/motions/f1_gmr_csv \
  --output_dir robolab/data/motions/f1_lab \
  --headless --device cpu --overwrite
```

The converter expects `root_pos(3) + root_quat_xyzw(4) + dof_pos(28)`.
It preserves every joint value, queries the imported articulation for its
runtime joint order, and calculates `key_body_pos` using the F1 URDF.

The current F1 Isaac Lab articulation order is:

1. `left_hip_pitch_joint`
2. `right_hip_pitch_joint`
3. `waist_yaw_joint`
4. `left_hip_roll_joint`
5. `right_hip_roll_joint`
6. `waist_roll_joint`
7. `left_hip_yaw_joint`
8. `right_hip_yaw_joint`
9. `left_shoulder_pitch_joint`
10. `right_shoulder_pitch_joint`
11. `left_knee_joint`
12. `right_knee_joint`
13. `left_shoulder_roll_joint`
14. `right_shoulder_roll_joint`
15. `left_ankle_pitch_joint`
16. `right_ankle_pitch_joint`
17. `left_shoulder_yaw_joint`
18. `right_shoulder_yaw_joint`
19. `left_ankle_roll_joint`
20. `right_ankle_roll_joint`
21. `left_elbow_joint`
22. `right_elbow_joint`
23. `left_wrist_roll_joint`
24. `right_wrist_roll_joint`
25. `left_wrist_yaw_joint`
26. `right_wrist_yaw_joint`
27. `left_wrist_pitch_joint`
28. `right_wrist_pitch_joint`

## Motion playback

Play one converted motion in the Isaac Lab window:

```bash
python robolab/scripts/tools/retarget/play_f1_motion_pkl.py \
  --motion_file robolab/data/motions/f1_lab/36_01.pkl \
  --loop
```

Play all pickles in the directory in filename order and repeat the playlist:

```bash
python robolab/scripts/tools/retarget/play_f1_motion_pkl.py \
  --motion_dir robolab/data/motions/f1_lab \
  --all --loop
```

Use `--names 36_01 36_11 turn_l turn_r` after `--motion_dir` to play only a
chosen ordered subset. The player applies the stored root pose and 28 joint
positions directly; it visualizes the reference data rather than a trained
policy tracking the motion.


- `pipeline/`: Single-motion generation, 515-motion batch processing, MuJoCo schemas, trajectory post-processing, and validation utilities.
- `general_motion_retargeting/`: Production implementation of the GMR / Mink retargeting solver.
- `configs/`: Production configuration files for the 6 humanoid robots.
- `general_motion_retargeting/ik_configs/`: SMPL-X inverse kinematics (IK) configuration files for each robot.
- `assets/`: Robot URDF/MJCF models and symbolic links to SMPL-X body models.

## Supported Robots

```text
unitree_g1
booster_t1
stanford_toddy
fourier_n1
hightorque_hi
engineai_pm01
```

## Symbolic Links

Set up the required symlink to the SMPL-X body models:

```bash
assets/body_models/smplx -> /data/smpl_models/smplx
```

## Environment & Dependencies

Verified environment:

```bash
/home/dell/anaconda3/envs/myenv/bin/python
```

Core dependencies include:
- `NumPy`, `SciPy`, `PyTorch`
- `SMPL-X`, `MuJoCo`, `Mink`, `DAQP`
- `PyYAML`, `Matplotlib`, `Rich`, `tqdm`

All required packages are documented in `requirements.txt`.

## Default Paths & I/O

```text
Candidate Manifest: /data/h2o_data_engine/selection/artifacts/v4/candidate_v4.jsonl
Canonical Motions:  /data/h2o_data_engine/work/canonical/motions/<motion_id>/source/
PLY Files:          /data/h2o_data_engine/work/ply/<motion_id>/
Output Directory:   /data/h2o_data_engine/work/robots/
```

If the canonical data, PLY files, or output directory reside on external storage, update the `paths` section in the corresponding `configs/robot_generation_*.yaml`. 

> **Note:** PLY files are strictly utilized to verify temporal alignment between the 15 FPS robot trajectories and visual frames; they are not involved in IK solving.

## Single-Motion Generation

Run the following command from the repository root:

```bash
cd /data/h2o_data_engine

MPLCONFIGDIR=/tmp/robot-retarget-mpl \
/home/dell/anaconda3/envs/myenv/bin/python \
  robot_retargeting/pipeline/generate_one.py \
  --config robot_retargeting/configs/robot_generation_g1.yaml \
  --motion-id <motion_id> \
  --resume
```

To target a different robot, specify its corresponding config file. Pass `--overwrite` only when explicitly intending to regenerate and overwrite existing results.

## Batch Generation (Per Robot)

Process all motions for a given robot:

```bash
MPLCONFIGDIR=/tmp/robot-retarget-mpl \
/home/dell/anaconda3/envs/myenv/bin/python \
  robot_retargeting/pipeline/generate_batch.py \
  --config robot_retargeting/configs/robot_generation_g1.yaml \
  --all
```

Execute this command sequentially for each of the 6 configuration files to retarget motions across all robots. 

The batch script implements automatic **resume semantics**: complete and identity-verified results are automatically validated and skipped, while failed cases are logged under `<output_root>/failures/`.

## Result Validation

Validate all MuJoCo frames for a single motion output directory:

```bash
/home/dell/anaconda3/envs/myenv/bin/python \
  robot_retargeting/pipeline/validate_output.py \
  --robot-dir work/robots/motions/<motion_id>/robots/unitree_g1 \
  --all-mujoco-frames \
  --write-report
```

Validate all existing results listed in a robot inventory manifest:

```bash
/home/dell/anaconda3/envs/myenv/bin/python \
  robot_retargeting/pipeline/generate_batch.py \
  --config robot_retargeting/configs/robot_generation_g1.yaml \
  --all \
  --validate-only \
  --all-mujoco-frames
```

## Output Structure

```text
work/robots/
  robot_schemas/<robot_id>.json
  manifests/robot_generation_inventory_<robot_id>.jsonl
  manifests/robot_generation_summary_<robot_id>.json
  motions/<motion_id>/robots/<robot_id>/
    trajectory_30fps.npz
    trajectory_15fps.npz
    diagnostics.npz
    metadata.json
    quality.json
    validation.json
```

The 15 FPS trajectory strictly samples even indices (`0, 2, 4, ...`) of the 30 FPS trajectory, sharing the exact visual frame IDs with the Avatar PLY sequences, RGB images, TIFF depth maps, and foreground masks.

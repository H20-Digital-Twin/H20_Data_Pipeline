# Dataset Generation Pipeline | H2O Method


This repository contains the official, automated dataset synthesis pipeline for **H2O: An Automated Digital Twin Method for Human-to-Humanoid Visual-Motor Synthesis**. The engine bridges human motion understanding and humanoid robotics by performing semantic motion filtering, canonical SMPL-X normalization, 3D implicit avatar geometry extraction, multi-view multi-modal photorealistic rendering, and kinematic motion retargeting across six diverse humanoid robot platforms.

> **Anonymous Peer Review Notice:** This codebase is structured to comply with strict **Double-Blind Review** protocols.
---

## Table of Contents
- [Pipeline Architecture](#pipeline-architecture)
- [Dataset Audit & Frozen Core Set ](#dataset-audit--frozen-core-set)
- [Repository Layout](#repository-layout)
- [Environment Setup](#environment-setup)
- [Data Directory Layout](#data-directory-layout)
- [Step-by-Step Pipeline Execution](#step-by-step-pipeline-execution)
  - [1. Motion Selection & Semantic Filtering](#1-motion-selection--semantic-filtering)
  - [2. Canonical Motion Synthesis (30 FPS)](#2-canonical-motion-synthesis-30-fps)
  - [3. Avatar Assignment](#3-avatar-assignment)
  - [4. 3D Avatar Geometry Generation (15 FPS PLY)](#4-3d-avatar-geometry-generation-15-fps-ply)
  - [5. Multi-View Multi-Modal Rendering](#5-multi-view-multi-modal-rendering)
  - [6. Humanoid Robot Motion Retargeting](#6-humanoid-robot-motion-retargeting)
- [One-Click Automated Generation](#one-click-automated-generation)
- [Data Specifications & Schemas](#data-specifications--schemas)
- [Visualization Utilities](#visualization-utilities)
- [License & Attribution](#license--attribution)

---

## Pipeline Architecture

The H2O dataset synthesis engine operates as an integrated multi-stage pipeline designed for scalable digital twin creation:

```text
       Raw Human Motion Library (SMPL-X Stage-II) + BABEL Semantic Labels
                                       │
                                       ▼
                  Stage 1: Semantic & Kinematic Filtering
                   ├── V2: Dynamic Quality & Semantic Pre-filter
                   ├── V3: Neutral FK Verification, Deduplication & Quotas
                   └──  BABEL Core Set Audit (515 Verified Sequences)
                                       │
                                       ▼
                  Stage 2: Canonical Motion Representation
                   └── Uniform 30 FPS SMPL-X Trajectories + Frame Maps
                                       │
                         ┌─────────────┴─────────────┐
                         ▼                           ▼
            Stage 3 & 4: Visual Engine     Stage 5: Kinematic Engine
            ├── Deterministic Avatar Alloc  ├── 6 Humanoid Robot Targets
            ├── 15 FPS 3D Mesh Engine       │   (G1, T1, Toddy, N1, HI, PM01)
            └── 4-Cam Rendering             └── Inverse Kinematics Synthesis
                (RGB/Depth/Mask)                (30 FPS + Aligned 15 FPS)
```

1. **Semantic & Kinematic Selection:** Filters raw human motion inputs using BABEL semantic tags, kinematic constraints, and source distribution caps.
2. **Canonical Sequence Generation:** Normalizes poses into a global frame of reference at 30 FPS.
3. **Implicit 3D Geometry Extraction:** Synthesizes dense 3D surface meshes (PLY) at 15 FPS using neural implicit avatars.
4. **Multi-View Multi-Modal Rendering:** Renders 4 synchronized fixed cameras producing aligned RGB, float32 metric depth, and segmentation masks.
5. **Humanoid Motion Retargeting:** Translates canonical motions onto six distinct physical humanoid kinematics platforms.

---

## Dataset Audit & Frozen Core Set 

The distribution across motion sub-collections is summarized below:

| Mocap Sub-collection | Sequence Count | Duration (seconds) |
| :--- | ---: | ---: |
| **Sub-collection A (ACCAD)** | 76 | 416.76 |
| **Sub-collection B (BMLmovi)** | 154 | 857.88 |
| **Sub-collection C (BMLrub)** | 144 | 1,018.71 |
| **Sub-collection D (CMU)** | 114 | 1,018.20 |
| **Sub-collection E (DFaust)** | 27 | 84.87 |




### Derivation Lineage
```text
8,345 Raw Motions 
  └──> V2 Base Pool (5,069)
         └──> V3 Kinematic Pool (897)
                └──> V4 Semantic Pruning (-50 hard rejects, -69 unverified semantics -> 778)
                       └──> Source Quotas (30% max duration per source sub-collection -> 515)
```

---

## Repository Layout

```text
├── selection/                  # Stage 1: Filtering & Selection Engine
│   ├── tools/                  # Script utilities for V2/V3/V4 selection
│   ├── configs/                # Hyperparameters, thresholds, and quota rules
│   └── artifacts/              # Frozen manifest snapshots and audit logs
├── generation/                 # Stage 2-4: Visual & Geometry Processing
│   ├── canonical/              # 30 FPS SMPL-X normalization scripts
│   ├── render/                 # Multi-camera rendering & sensor simulation
│   └── visualize_depth_tiff.py # Metric depth inspection tool
├── robot_retargeting/          # Stage 5: Multi-Robot Kinematic Retargeting
│   ├── configs/                # Target robot URDF/IK configurations
│   └── pipeline/               # Batch solver routines for 6 humanoid models
├── vendor/                     # Self-contained third-party neural avatar code & models
│   └── avatar_engine/          # Neural implicit surface synthesis code & SMPL-X models
├── assets/                     # Digital Twin Avatar assets & meta checkpoints
│   └── avatars/                # 6 simplified avatar directories
├── scripts/                    # Master execution bash scripts
└── work/                       # Workspace for generated outputs (created dynamically)
```

---

## Environment Setup

The pipeline requires CUDA-enabled PyTorch and hardware rendering libraries. We recommend using two isolated Conda environments to prevent dependency conflicts between visual synthesis and numerical IK solvers.

### Prerequisites & C++ Extensions
- Linux (x86_64)
- CUDA 11.8 / 12.1 compatible GPU (>= 16GB VRAM recommended)
- GCC / G++ 7.5+

### 1. Vision & Mesh Processing Environment (`py38_vision`)

```bash
# Create Environment
conda create -n py38_vision python=3.8 -y
conda activate py38_vision

# Install Core Dependencies
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install pytorch-lightning==1.9.5 hydra-core==1.3.2 imageio tifffile opencv-python

# Install PyTorch3D
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.4"

# Install repository dependencies
pip install -r requirements.txt
```

> **C++ Extension Notice:** The pre-compiled mesh extraction extension (`libmise`) included in `vendor/avatar_engine/` is built for Linux x86-64 / CPython 3.8. If changing Python versions, recompile the extension via:
> ```bash
> cd vendor/avatar_engine/code/libmise && python setup.py build_ext --inplace
> ```

### 2. Robot Retargeting Environment (`py38_robot`)

```bash
conda create -n py38_robot python=3.8 -y
conda activate py38_robot

pip install numpy scipy pinocchio transformation trimesh matplotlib pyyaml
```

---

## Data Directory Layout

Before running the dataset generation scripts, organize your raw inputs according to the default paths or set environment overrides:

```text
./data/
├── source_motions/             # Raw SMPL-X Motion Sequences
│   └── smpl-x-n/
└── semantic_labels/            # BABEL Semantic Annotation Files
    ├── train.json
    ├── val.json
    └── test.json
```

If your datasets are stored in alternative locations, set environment variables:

```bash
export SOURCE_MOTION_ROOT=/path/to/your/mocap_data
export BABEL_ROOT=/path/to/your/babel_annotations
```

The current production deployment uses the following H2O names:

```text
/data/h2o_data_engine                         # self-contained generation engine
/data/avatar_plys/h2o_avatar_ply             # 15 FPS human avatar meshes
/data/avatar_plys/h2o_human_multiview        # aligned 4-view RGB/depth/mask data
/data/avatar_plys/h2o_humanoid_motion        # six-robot retargeted trajectories
```

These replace the former `robot_dataset` and `stage2_*` deployment directory
names. Internal `stage2_multiview_*` schema/version strings remain unchanged so
existing render metadata stays resume-compatible.

---

## Step-by-Step Pipeline Execution

Run pre-flight validation to confirm runtime paths and CUDA availability:

```bash
scripts/preflight.sh
```

### 1. Motion Selection & Semantic Filtering
The repository contains pre-computed manifests. To reproduce the selection procedure from scratch:

```bash
CONFIRM_SELECTION_REBUILD=YES scripts/build_candidates_all.sh
```
*Note: Stage V3 computes Neutral SMPL-X Forward Kinematics across candidate frames to eliminate self-collisions and anomalous poses.*

### 2. Canonical Motion Synthesis (30 FPS)
Converts raw motions into canonical SMPL-X format aligned at 30 FPS.

```bash
CONFIRM_FULL_RUN=YES scripts/build_canonical_all.sh
```
**Output Directory:** `work/canonical/motions/<motion_id>/source/`
- `canonical_smplx_30fps.npz`: Complete sequence pose and translation vectors.
- `frame_map_15fps.npz`: Subsampling indices mapping 30 FPS to 15 FPS visual tracks.
- `canonical_metadata.json`: Sequence length, source metadata, and semantic tags.

### 3. Avatar Assignment
Deterministically maps the 515 sequences to six digital human avatars (`00025_scan`, `00034_rgbd`, `00085_scan`, `00020_scan`, `00027_scan`, `00087_rgbd`):

```bash
scripts/prepare_avatar_assignment.sh
```

### 4. 3D Avatar Geometry Generation (15 FPS PLY)
Runs implicit avatar surface reconstruction to produce dense 3D PLY meshes at 15 FPS:

```bash
CONFIRM_FULL_RUN=YES scripts/generate_ply_all.sh
```
- Outputs PLY files to `work/ply/<motion_id>/*.ply`.
- Check batch progress using: `scripts/ply_status.sh`
- Supports custom storage paths via `OUTPUT_ROOT=/path/to/storage scripts/generate_ply_all.sh`.

### 5. Multi-View Multi-Modal Rendering
Simulates a 4-camera studio array to produce multi-modal visual tracks (RGB, metric depth, masks):

```bash
CONFIRM_MULTIVIEW_FULL_RUN=YES scripts/render_multiview_all.sh
```

**Output Structure (`work/multiview/<motion_id>/`):**
```text
├── cameras.npz                  # Extrinsics & Intrinsics for cam_000..cam_003
├── cam_000/
│   ├── rgb/000000.png          # 512x512 uint8 RGB
│   ├── depth/000000.tiff        # 512x512 float32 Depth (in meters)
│   └── mask/000000.png         # 512x512 uint8 Binary Mask (0 or 255)
└── ... [cam_001, cam_002, cam_003]
```

### 6. Humanoid Robot Motion Retargeting
Retargets canonical SMPL-X motions onto six target humanoid platforms using non-linear IK optimization.

```bash
conda activate py38_robot

# Example: Generate Unitree G1 trajectories
python robot_retargeting/pipeline/generate_batch.py \
  --config robot_retargeting/configs/robot_generation_g1.yaml \
  --all
```

To execute for other robot targets, replace the configuration flag:

- **Booster T1:** `robot_generation_booster_t1.yaml`
- **Stanford Toddy:** `robot_generation_stanford_toddy.yaml`
- **Fourier N1:** `robot_generation_fourier_n1.yaml`
- **HighTorque HI:** `robot_generation_hightorque_hi.yaml`
- **EngineAI PM01:** `robot_generation_engineai_pm01.yaml`

---

## One-Click Automated Generation

To run the visual generation pipeline end-to-end:

```bash
CONFIRM_FULL_PIPELINE=YES scripts/run_full_pipeline.sh
```

To include full selection rebuilding from raw inputs:

```bash
REBUILD_SELECTION=YES \
CONFIRM_FULL_PIPELINE=YES \
scripts/run_full_pipeline.sh
```

---

## Data Specifications & Schemas

### Visual Renderings
- **RGB Images:** `512 x 512` uint8 PNG files with solid white backgrounds.
- **Depth Maps:** `512 x 512` float32 single-channel TIFF format. Values encode exact Euclidean distance in **meters** (Background value: `0.0`).
- **Segmentation Masks:** `512 x 512` uint8 PNG images (`255` = Foreground subject, `0` = Background).
- **Camera Extrinsics/Intrinsics:** Stored in `cameras.npz` containing 4x4 projection matrices, focal lengths, and principal points per camera.

### Robot Trajectories
- **Sampling Rate:** Primary trajectory provided at 30 FPS, alongside a synchronized 15 FPS downsampled subset aligned with visual mesh frames.
- **Format:** `.npz` files containing joint positions ($q$), joint velocities ($\dot{q}$), root transformation, and end-effector cartesian tracks.

---

## Visualization Utilities

Because metric depth files are stored as raw 32-bit floating-point TIFF matrices, standard viewer applications may display them incorrectly. We provide a preview renderer to visualize aligned RGB, Depth, and Mask samples side-by-side:

```bash
python generation/visualize_depth_tiff.py \
  work/multiview/<motion_id>/cam_000/depth/000000.tiff \
  --rgb work/multiview/<motion_id>/cam_000/rgb/000000.png \
  --mask work/multiview/<motion_id>/cam_000/mask/000000.png \
  --output /tmp/multimodal_preview.png
```

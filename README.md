# Orthodontic Tooth Trajectory Project (Stage 0: Pseudo-Staging · Stage 1: Diffusion)

> **Status:** Work in progress (WIP).  
> This repository currently provides **Stage 0: pseudo-staging / pseudo-label generation** from pre/post states.  
> **Stage 1 (diffusion-based trajectory model)** will be added in this same repo.

This project studies orthodontic tooth movement trajectory learning when **only pre-treatment (Ori)** and **post-treatment (Final)** states are available (no real intermediate supervision).  
To enable model training, we first synthesize **pseudo intermediate trajectories** (multi-modal) and evaluate them with collision-aware metrics.

---

## Why pseudo-staging matters (Stage 0)
Real intermediate tooth poses are often missing. Stage 0 generates **pseudo supervision** to support downstream learning.

Given:
- `ori/{JAW}_Ori.stl` — full mesh at pre-treatment
- `ori/{JAW}_Ori.json` — per-tooth segmentation vertices (pre)
- `final/{JAW}_Final.json` — per-tooth segmentation vertices (post)

Stage 0 produces:
- multiple pseudo trajectories (`traj_XX/step_YYY.stl`)
- `poses.json` metadata (per-tooth pose per step)

---

## Key features (Stage 0)
### 1) Per-tooth registration (pre → post)
- Estimates rigid transform for each tooth using **Kabsch** (when applicable) with **ICP fallback**

### 2) Multi-trajectory pseudo staging (augmentation)
- Generates **NUM_TRAJ** trajectories by sampling **per-tooth schedules**
  - start delay + gamma time-warp
  - optional **posterior → anterior** group delay (molar-first tendency)

### 3) Extraction-aware topology handling
- Detects extracted teeth via **pre-only tooth IDs**
- Removes corresponding faces after an early step (configurable)

### 4) Light collision damping
- Reduces only severe near-collisions without iterative freezing
- **Does not modify the final step** (final reached)

## output sample:

![TAAP Samples](stage0_output/1(traj_5 version).mp4)

---



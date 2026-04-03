#ifndef MOL_DIFFUSION_CONFIG_H
#define MOL_DIFFUSION_CONFIG_H

// =============================================================================
//  Default model parameters — override any of these at compile time, e.g.:
//    nvcc -DMODEL_FEAT_DIM=256 -DTRAIN_BATCH_SIZE=64 ...
//
//  ModelConfig and TrainConfig use these as their default member values, so
//  default-constructed configs automatically pick up the defines.  Fields set
//  explicitly at runtime (e.g. from CLI args) still override as before.
// =============================================================================

// ── Diffusion schedule (also consumed directly by DiffusionModule kernels) ────
#ifndef DIFF_T
#  define DIFF_T                1000      // total diffusion timesteps T
#endif
#ifndef DIFF_T_INFER
#  define DIFF_T_INFER           500      // timestep used at inference
#endif
#ifndef DIFF_NOISE_SCALE
#  define DIFF_NOISE_SCALE       0.15f    // noise magnitude during training
#endif

// ── ModelConfig defaults ──────────────────────────────────────────────────────
#ifndef MODEL_N_TIMESTEPS
#  define MODEL_N_TIMESTEPS      DIFF_T   // kept in sync with DIFF_T by default
#endif
#ifndef MODEL_BETA_START
#  define MODEL_BETA_START       1e-4f
#endif
#ifndef MODEL_BETA_END
#  define MODEL_BETA_END         2e-2f
#endif
#ifndef MODEL_COND_DIM
#  define MODEL_COND_DIM         1024      // PointNet++ output / conditioning dim
#endif
#ifndef MODEL_PC_N_POINTS
#  define MODEL_PC_N_POINTS      1024      // point cloud size fed to the encoder
#endif
#ifndef MODEL_FEAT_DIM
#  define MODEL_FEAT_DIM         512      // EGNN node feature dimension
#endif
#ifndef MODEL_TIME_DIM
#  define MODEL_TIME_DIM         256       // sinusoidal timestep embedding dim
#endif
#ifndef MODEL_N_LAYERS
#  define MODEL_N_LAYERS         8        // EGNN message-passing layers
#endif
#ifndef MODEL_RADIUS
#  define MODEL_RADIUS           5.0f     // radius-graph cutoff (Angstroms)
#endif
#ifndef MODEL_N_ATOM_TYPES
#  define MODEL_N_ATOM_TYPES     11       // must match N_ATOM_TYPES / ATOM_SYMBOLS
#endif

// ── TrainConfig defaults ──────────────────────────────────────────────────────
#ifndef TRAIN_EPOCHS
#  define TRAIN_EPOCHS           200
#endif
#ifndef TRAIN_BATCH_SIZE
#  define TRAIN_BATCH_SIZE       32
#endif
#ifndef TRAIN_LR
#  define TRAIN_LR               1e-4f
#endif
#ifndef TRAIN_MIN_LR
#  define TRAIN_MIN_LR           1e-6f
#endif
#ifndef TRAIN_WEIGHT_DECAY
#  define TRAIN_WEIGHT_DECAY     1e-4f
#endif
#ifndef TRAIN_GRAD_CLIP
#  define TRAIN_GRAD_CLIP        1.0f
#endif
#ifndef TRAIN_WARMUP_STEPS
#  define TRAIN_WARMUP_STEPS     1000
#endif
#ifndef TRAIN_EMA_DECAY
#  define TRAIN_EMA_DECAY        0.9999f
#endif
#ifndef TRAIN_NUM_WORKERS
#  define TRAIN_NUM_WORKERS      4
#endif
#ifndef TRAIN_SAVE_EVERY
#  define TRAIN_SAVE_EVERY       5
#endif
#ifndef TRAIN_LOG_EVERY
#  define TRAIN_LOG_EVERY        50
#endif
#ifndef TRAIN_SEED
#  define TRAIN_SEED             42
#endif

#endif // MOL_DIFFUSION_CONFIG_H

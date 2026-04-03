#ifndef MOL_DIFFUSION_MODEL_H
#define MOL_DIFFUSION_MODEL_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <string>
#include <tuple>

#include "mol_diffusion_config.h"
#include "mol_diffusion_types.h"

namespace mol_diffusion {

// ─────────────────────────────────────────────────────────────────────────────
// Weight tensors for a single EGNN layer (device memory)
// ─────────────────────────────────────────────────────────────────────────────
struct EGNNLayerWeights {
    // Edge MLP  phi_e: [D_in_e, D_feat]
    float* W_e = nullptr; float* b_e = nullptr;
    // Node MLP  phi_h: [2*D_feat, D_feat]
    float* W_h = nullptr; float* b_h = nullptr;
    // Coord net phi_x: [D_feat, 1]
    float* W_x = nullptr; float* b_x = nullptr;
    // LayerNorm  gamma, beta
    float* gamma = nullptr; float* beta_ln = nullptr;

    int D_feat, D_time, D_in_e;

    void allocate(int feat, int time_dim);
    void free();
    void xavier_init(unsigned seed);
};

// ─────────────────────────────────────────────────────────────────────────────
// PointNet++ encoder weights  (3 set-abstraction levels + 2 FC layers)
// ─────────────────────────────────────────────────────────────────────────────
struct PointNetWeights {
    // SA level 1: in=(3), out=128
    float* W_sa1 = nullptr; float* b_sa1 = nullptr;
    // SA level 2: in=(128), out=256
    float* W_sa2 = nullptr; float* b_sa2 = nullptr;
    // SA level 3: in=(256), out=1024
    float* W_sa3 = nullptr; float* b_sa3 = nullptr;
    // FC: 1024→512
    float* W_fc1 = nullptr; float* b_fc1 = nullptr;
    // FC: 512→cond_dim
    float* W_fc2 = nullptr; float* b_fc2 = nullptr;

    void allocate(int cond_dim);
    void free();
    void xavier_init(unsigned seed);
};

// ─────────────────────────────────────────────────────────────────────────────
// Output head weights
// ─────────────────────────────────────────────────────────────────────────────
struct HeadWeights {
    // Position head: feat → feat → 3
    float* W_p1 = nullptr; float* b_p1 = nullptr;
    float* W_p2 = nullptr; float* b_p2 = nullptr;
    // Type head: feat → feat → n_atom_types
    float* W_t1 = nullptr; float* b_t1 = nullptr;
    float* W_t2 = nullptr; float* b_t2 = nullptr;

    void allocate(int feat_dim, int n_atom_types);
    void free();
    void xavier_init(unsigned seed);
};

// ─────────────────────────────────────────────────────────────────────────────
// Model hyper-parameters (all in one struct)
// ─────────────────────────────────────────────────────────────────────────────
struct ModelConfig {
    // Diffusion schedule
    int   n_timesteps  = MODEL_N_TIMESTEPS;
    float beta_start   = MODEL_BETA_START;
    float beta_end     = MODEL_BETA_END;

    // PointNet++ encoder
    int   cond_dim     = MODEL_COND_DIM;
    int   pc_n_points  = MODEL_PC_N_POINTS;

    // EGNN denoiser
    int   feat_dim     = MODEL_FEAT_DIM;
    int   time_dim     = MODEL_TIME_DIM;
    int   n_layers     = MODEL_N_LAYERS;
    float radius       = MODEL_RADIUS;    // Angstroms

    // Atom types
    int   n_atom_types = MODEL_N_ATOM_TYPES;
};

// ─────────────────────────────────────────────────────────────────────────────
// Training hyper-parameters
// ─────────────────────────────────────────────────────────────────────────────
struct TrainConfig {
    int         epochs        = TRAIN_EPOCHS;
    int         batch_size    = TRAIN_BATCH_SIZE;
    float       lr            = TRAIN_LR;
    float       min_lr        = TRAIN_MIN_LR;
    float       weight_decay  = TRAIN_WEIGHT_DECAY;
    float       grad_clip     = TRAIN_GRAD_CLIP;
    int         warmup_steps  = TRAIN_WARMUP_STEPS;
    float       ema_decay     = TRAIN_EMA_DECAY;
    int         num_workers   = TRAIN_NUM_WORKERS;
    bool        use_amp       = true;
    int         save_every    = TRAIN_SAVE_EVERY;
    int         log_every     = TRAIN_LOG_EVERY;
    int         seed          = TRAIN_SEED;
    std::string ckpt_dir      = "checkpoints";
    std::string log_dir       = "runs";
    std::string resume        = "";
};

// ─────────────────────────────────────────────────────────────────────────────
// Noise schedule (precomputed on GPU)
// ─────────────────────────────────────────────────────────────────────────────
struct NoiseSchedule {
    float* d_betas = nullptr;   // [T]
    float* d_alphas = nullptr;   // [T]
    float* d_alpha_bar = nullptr;   // [T]
    float* d_sqrt_ab = nullptr;   // [T]  sqrt(alpha_bar)
    float* d_sqrt_1mab = nullptr;   // [T]  sqrt(1 - alpha_bar)
    int    T = 0;

    void allocate(int timesteps, float beta_start, float beta_end,
        cudaStream_t stream = 0);
    void free();
};

// Forward declarations (opaque pointers to hide implementation details)
struct DiffusionModule;

// ─────────────────────────────────────────────────────────────────────────────
// MolDiffusion model class
// ─────────────────────────────────────────────────────────────────────────────
class MolDiffusion {
public:
    explicit MolDiffusion(const ModelConfig& cfg);
    ~MolDiffusion();

    // Disable copy
    MolDiffusion(const MolDiffusion&) = delete;
    MolDiffusion& operator=(const MolDiffusion&) = delete;

    // ── Training ──────────────────────────────────────────────────────────────

    /// Compute loss for a batch:
    ///   pc        [B, N_pc, 3]  point clouds  (host)
    ///   pos       [N_tot, 3]    clean positions (host)
    ///   types     [N_tot]       clean atom types (host)
    ///   batch     [N_tot]       molecule index per atom (host)
    ///   Returns: (loss_total, loss_pos, loss_type)
    std::tuple<float, float, float> compute_loss(
        const std::vector<PointCloud>& pcs,
        const std::vector<Molecule>& mols);

    // ── Inference ─────────────────────────────────────────────────────────────

    /// Generate one molecule per point cloud in the batch.
    /// n_atoms:  number of atoms to generate per molecule
    std::vector<Molecule> sample(
        const std::vector<PointCloud>& pcs,
        int n_atoms = 20);

    // ── I/O ───────────────────────────────────────────────────────────────────
    void save(const std::string& path) const;
    void load(const std::string& path);

    // ── Accessors for trainer (gradient update) ───────────────────────────────
    std::vector<float*> parameters();          // all device weight ptrs
    std::vector<size_t> parameter_sizes();     // element counts

    const ModelConfig& config() const { return cfg_; }

    // Returns the raw DiffusionModule pointer (void* to avoid header dependency).
    // Trainer casts this to DiffusionModule* to call diff_zero_grads each step.
    void* denoiser_opaque() { return denoiser_opaque_; }

private:
    ModelConfig cfg_;

    std::vector<float*> param;
    std::vector<size_t> param_size;

    // GPU resources
    cudaStream_t   stream_ = 0;
    cublasHandle_t cublas_ = nullptr;

    // Noise schedule
    NoiseSchedule schedule_;

    // Learned diffusion denoiser (DiffusionModule defined in mol_diffusion.cu §15).
    // Held as a void* to avoid exposing DMatrix/DiffusionModule in the header;
    // cast back to DiffusionModule* in the .cu where the full type is known.
    void* denoiser_opaque_ = nullptr;

    // Weights
    PointNetWeights  pn_weights_;
    std::vector<EGNNLayerWeights> egnn_weights_;
    HeadWeights      head_weights_;

    // Scratch buffers (re-used across calls)
    struct Scratch {
        // PointNet
        float* d_pc = nullptr;   // [B*N_pc, 3]
        float* d_cent1 = nullptr;   // [B*512, 3]
        float* d_feat1 = nullptr;   // [B*512, 128]
        float* d_cent2 = nullptr;   // [B*128, 3]
        float* d_feat2 = nullptr;   // [B*128, 256]
        float* d_feat3 = nullptr;   // [B*1,   1024]
        float* d_cond = nullptr;   // [B, cond_dim]
        int* d_fps1 = nullptr;
        int* d_fps2 = nullptr;
        int* d_ball1 = nullptr;
        int* d_ball2 = nullptr;
        int* d_ball3 = nullptr;
        float* d_fps_dist = nullptr;
        float* d_fc_tmp = nullptr;     // [B, 1024]  temp buffer for encode_pc linear layers (avoids in-place aliasing)
        // Molecule
        float* d_pos = nullptr;   // [N_tot, 3]
        float* d_xt = nullptr;
        float* d_eps = nullptr;
        float* d_noise = nullptr;
        int* d_types = nullptr;
        int* d_types_t = nullptr;
        int* d_t_atom = nullptr;   // [N_tot]
        int* d_batch = nullptr;
        int* d_mol_off = nullptr;   // [B+1]
        // EGNN
        float* d_h = nullptr;
        float* d_h_agg = nullptr;
        float* d_h_tmp = nullptr;
        float* d_t_emb = nullptr;
        float* d_cond_atom = nullptr;
        float* d_msg = nullptr;
        int* d_edge_src = nullptr;
        int* d_edge_dst = nullptr;
        int* d_n_edges = nullptr;
        // Outputs
        float* d_pred_pos = nullptr;
        float* d_pred_type = nullptr;
        // Loss
        float* d_loss = nullptr;

        int allocated_N = 0;
        int allocated_B = 0;
        int allocated_E = 0;

        void ensure(int N_tot, int B, int N_pc,
            int cond_dim, int feat_dim, int time_dim,
            int n_atom_types, int max_edges);
        void free_all();
    } scratch_;

    // ── Private helpers ────────────────────────────────────────────────────────

    /// Encode point clouds [B, N_pc, 3] → [B, cond_dim]
    void encode_pc(const std::vector<PointCloud>& pcs);

    /// Run EGNN denoiser forward pass
    void run_denoiser(int N_tot, int B, const int* d_batch_offsets);

    /// Upload a batch of molecules to GPU scratch buffers
    void upload_molecules(const std::vector<Molecule>& mols,
        std::vector<int>& offsets);

    void init_weights();
};

} // namespace mol_diffusion

#endif // MOL_DIFFUSION_MODEL_H

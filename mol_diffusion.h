
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>

#include <stdio.h>
#include <math.h>
#include <cstdint>
#include <float.h>

#include <getopt.h>

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

#include <fstream>
#include <sstream>
#include <iostream>

#include <stdexcept>
#include <filesystem>

#include <random>
#include <numeric>
#include <algorithm>
#include <Eigen/Dense>

#ifndef _GLIBCXX_USE_DEPRECATED
#  define _GLIBCXX_USE_DEPRECATED 1
#endif

#include <openbabel/mol.h>
#include <openbabel/obconversion.h>
#include <openbabel/forcefield.h>

#include <openbabel/mol.h>
#include <openbabel/atom.h>
#include <openbabel/bond.h>
#include <openbabel/obconversion.h>
#include <openbabel/forcefield.h>
#include <openbabel/elements.h>
#include <openbabel/obiter.h>
#include <openbabel/generic.h>

using namespace OpenBabel;


// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────
#define CUDA_CHECK(err) \
    if ((err) != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d  %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    }

#define CUDA_CHECK2(err) \
    if ((err) != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(err)); }
#define CUDA_CHECK_LAST()  CUDA_CHECK(cudaGetLastError())

static __device__ __forceinline__ float clamp(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}


//mol_diffusion
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
// Atom type mapping (matches Python reference code)
// ─────────────────────────────────────────────────────────────────────────────
constexpr int N_ATOM_TYPES = 10;
static const char* ATOM_SYMBOLS[N_ATOM_TYPES] = {
    "H", "C", "N", "O", "F", "S", "Cl", "Br", "P", "I"
};
static const int ATOM_VALENCE[N_ATOM_TYPES] = {
    1,   4,   3,   2,   1,   2,    1,    1,   5,   1
};

// ─────────────────────────────────────────────────────────────────────────────
// Point cloud
// ─────────────────────────────────────────────────────────────────────────────
struct PointCloud {
    std::vector<float> xyz;  // [N*3]  row-major (x,y,z)
    int N = 0;

    float* data() { return xyz.data(); }
    const float* data() const { return xyz.data(); }
    void resize(int n) { N = n; xyz.resize(n * 3, 0.f); }

    Eigen::Vector3f point(int i) const {
        return { xyz[i * 3], xyz[i * 3 + 1], xyz[i * 3 + 2] };
    }
    void normalize();   // center + scale to unit sphere
};

// ─────────────────────────────────────────────────────────────────────────────
// Molecule (clean or noisy)
// ─────────────────────────────────────────────────────────────────────────────
struct Molecule {
    std::vector<float>    pos;        // [N_atoms * 3]
    std::vector<int32_t>  atom_types; // [N_atoms]
    int N = 0;

    void resize(int n) {
        N = n;
        pos.assign(n * 3, 0.f);
        atom_types.assign(n, 0);
    }

    Eigen::Vector3f position(int i) const {
        return { pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2] };
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Training sample
// ─────────────────────────────────────────────────────────────────────────────
struct Sample {
    PointCloud  pc;
    Molecule    mol;
};


class MoleculeBuilder {
public:
    // ── Core builder ─────────────────────────────────────────────────────────

    /// Build an OBMol with 3-D coordinates and distance-inferred bonds.
    /// Caller owns the returned pointer (or use via process()).
    static OpenBabel::OBMol* build_obmol(const Molecule& mol,
        float bond_tolerance = 1.15f);

    /// Run MMFF94 (or UFF fallback) geometry optimisation in-place.
    /// Returns true if a force field was found and ran successfully.
    static bool relax(OpenBabel::OBMol& obmol,
        int         max_steps = 500,
        const char* ff_id = "MMFF94");

    /// Full pipeline: build → PerceiveBondOrders → AddHydrogens → relax.
    /// Returns nullptr if the molecule cannot be built.
    static std::unique_ptr<OpenBabel::OBMol> process(
        const Molecule& mol,
        bool  do_relax = true,
        float bond_tolerance = 1.15f);

    // ── Format writers ───────────────────────────────────────────────────────

    /// Write a batch to an SDF V2000 file.
    static void write_sdf(const std::vector<Molecule>& mols,
        const std::string& path,
        bool  do_relax = true,
        float bond_tolerance = 1.15f);

    /// Write a batch to a MOL2 (Tripos) file.
    static void write_mol2(const std::vector<Molecule>& mols,
        const std::string& path,
        bool  do_relax = true,
        float bond_tolerance = 1.15f);

    /// Write raw positions + symbols to XYZ (no bonds required).
    static void write_xyz(const std::vector<Molecule>& mols,
        const std::string& path);

    /// Canonical SMILES via OBConversion.  Returns "" on failure.
    static std::string to_smiles(const Molecule& mol,
        float bond_tolerance = 1.15f);

    /// Return true when OpenBabel successfully perceives bonds and all
    /// atom valences are chemically reasonable.
    static bool is_valid(const Molecule& mol,
        float bond_tolerance = 1.15f);

    // ── Helpers (public for unit tests) ─────────────────────────────────────

    static const char* atom_symbol(int type_idx);
    static float       cov_radius(int type_idx);

private:
    static void write_batch(const std::vector<Molecule>& mols,
        const std::string& path,
        const char* ob_format,
        bool  do_relax,
        float bond_tolerance);
};

// ─────────────────────────────────────────────────────────────────────────────
// Model hyper-parameters (all in one struct)
// ─────────────────────────────────────────────────────────────────────────────
struct ModelConfig {
    // Diffusion
    int   n_timesteps = 1000;
    float beta_start = 1e-4f;
    float beta_end = 2e-2f;

    // PointNet++ encoder
    int   cond_dim = 256;
    int   pc_n_points = 256;

    // EGNN denoiser
    int   feat_dim = 128;
    int   time_dim = 64;
    int   n_layers = 6;
    float radius = 5.0f;   // Angstroms

    // Atom types
    int   n_atom_types = N_ATOM_TYPES;
};

// ─────────────────────────────────────────────────────────────────────────────
// Training hyper-parameters
// ─────────────────────────────────────────────────────────────────────────────
struct TrainConfig {
    int   epochs = 200;
    int   batch_size = 32;
    float lr = 1e-4f;
    float min_lr = 1e-6f;
    float weight_decay = 1e-4f;
    float grad_clip = 1.0f;
    int   warmup_steps = 1000;
    float ema_decay = 0.9999f;
    int   num_workers = 4;
    bool  use_amp = true;
    int   save_every = 5;
    int   log_every = 50;
    int   seed = 42;
    std::string ckpt_dir = "checkpoints";
    std::string log_dir = "runs";
    std::string resume = "";
};

// ─────────────────────────────────────────────────────────────────────────────
// GPU tensor descriptor (owned by CUDA device memory)
// ─────────────────────────────────────────────────────────────────────────────
struct DeviceTensor {
    float* ptr = nullptr;
    int      rows = 0;
    int      cols = 0;
    size_t   bytes() const { return (size_t)rows * cols * sizeof(float); }
    void     free_gpu();
    void     alloc_gpu(int r, int c);
};

struct DeviceTensorI {
    int32_t* ptr = nullptr;
    int      n = 0;
    size_t   bytes() const { return (size_t)n * sizeof(int32_t); }
    void     free_gpu();
    void     alloc_gpu(int n_);
};



// ─────────────────────────────────────────────────────────────────────────────
// Dataset base class
// ─────────────────────────────────────────────────────────────────────────────
class Dataset {
public:
    virtual ~Dataset() = default;
    virtual size_t size() const = 0;
    virtual Sample& operator[](size_t idx) = 0;
    virtual void shuffle() = 0;
};

// ─────────────────────────────────────────────────────────────────────────────
// XYZ molecule → point cloud derivation helper
// ─────────────────────────────────────────────────────────────────────────────
inline PointCloud derive_pc(const Molecule& mol, int n_pc_points,
    float noise_std, std::mt19937& rng)
{
    std::normal_distribution<float> noise(0.f, noise_std);
    std::uniform_int_distribution<int> atom_idx(0, mol.N - 1);

    PointCloud pc;
    pc.resize(n_pc_points);

    // First: noisy copies of atom positions
    int pc_from_atoms = std::min(mol.N, n_pc_points);
    for (int i = 0; i < pc_from_atoms; ++i) {
        pc.xyz[i * 3 + 0] = mol.pos[i * 3 + 0] + noise(rng);
        pc.xyz[i * 3 + 1] = mol.pos[i * 3 + 1] + noise(rng);
        pc.xyz[i * 3 + 2] = mol.pos[i * 3 + 2] + noise(rng);
    }

    // Extra: interpolated surface points
    for (int i = pc_from_atoms; i < n_pc_points; ++i) {
        int a = atom_idx(rng), b = atom_idx(rng);
        float lam = std::uniform_real_distribution<float>(0.f, 1.f)(rng);
        pc.xyz[i * 3 + 0] = lam * mol.pos[a * 3 + 0] + (1 - lam) * mol.pos[b * 3 + 0] + noise(rng) * 0.3f;
        pc.xyz[i * 3 + 1] = lam * mol.pos[a * 3 + 1] + (1 - lam) * mol.pos[b * 3 + 1] + noise(rng) * 0.3f;
        pc.xyz[i * 3 + 2] = lam * mol.pos[a * 3 + 2] + (1 - lam) * mol.pos[b * 3 + 2] + noise(rng) * 0.3f;
    }
    return pc;
}

// ─────────────────────────────────────────────────────────────────────────────
// Parse a single .xyz file
// ─────────────────────────────────────────────────────────────────────────────
static const std::unordered_map<std::string, int>& get_sym_map() {
    static const std::unordered_map<std::string, int> m = {
        {"H",0},{"C",1},{"N",2},{"O",3},{"F",4},
        {"S",5},{"Cl",6},{"Br",7},{"P",8},{"I",9}
    };
    return m;
}

inline bool parse_xyz(const std::string& path, Molecule& mol)
{
    std::ifstream f(path);
    if (!f) return false;
    int n; f >> n;
    if (n <= 0 || n > 500) return false;
    std::string comment; std::getline(f >> std::ws, comment);
    mol.resize(n);
    for (int i = 0; i < n; ++i) {
        std::string sym; float x, y, z;
        f >> sym >> x >> y >> z;
        if (!f) return false;
        auto it = get_sym_map().find(sym);
        mol.atom_types[i] = (it != get_sym_map().end()) ? it->second : 1; // default C
        mol.pos[i * 3 + 0] = x;
        mol.pos[i * 3 + 1] = y;
        mol.pos[i * 3 + 2] = z;
    }
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// XYZDataset: loads all .xyz files from a directory
// ─────────────────────────────────────────────────────────────────────────────
class XYZDataset : public Dataset {
public:
    XYZDataset(const std::string& dir,
        int n_pc_points = 256,
        float noise_std = 0.3f,
        int seed = 42)
        : n_pc_points_(n_pc_points), noise_std_(noise_std), rng_(seed)
    {
        namespace fs = std::filesystem;
        for (auto& p : fs::directory_iterator(dir)) {
            if (p.path().extension() == ".xyz") {
                Molecule mol;
                if (parse_xyz(p.path().string(), mol)) {
                    PointCloud pc = derive_pc(mol, n_pc_points_, noise_std_, rng_);
                    samples_.push_back({ std::move(pc), std::move(mol) });
                }
            }
        }
        indices_.resize(samples_.size());
        std::iota(indices_.begin(), indices_.end(), 0);
        std::cout << "[XYZDataset] Loaded " << samples_.size() << " molecules from " << dir << "\n";
    }

    size_t size() const override { return samples_.size(); }
    Sample& operator[](size_t idx) override { return samples_[indices_[idx]]; }
    void shuffle() override { std::shuffle(indices_.begin(), indices_.end(), rng_); }

private:
    std::vector<Sample> samples_;
    std::vector<int>    indices_;
    int   n_pc_points_;
    float noise_std_;
    std::mt19937 rng_;
};

// ─────────────────────────────────────────────────────────────────────────────
// SyntheticDataset: random Gaussian molecules + derived point clouds
// ─────────────────────────────────────────────────────────────────────────────
class SyntheticDataset : public Dataset {
public:
    SyntheticDataset(int n_samples = 10000,
        int min_atoms = 5,
        int max_atoms = 30,
        int n_pc_points = 256,
        int n_atom_types = 10,
        float noise_std = 0.5f,
        int seed = 42)
        : rng_(seed)
    {
        std::normal_distribution<float> pos_dist(0.f, 1.f);
        std::uniform_int_distribution<int> n_atom_dist(min_atoms, max_atoms);
        std::uniform_int_distribution<int> type_dist(0, n_atom_types - 1);

        samples_.resize(n_samples);
        for (auto& s : samples_) {
            int n = n_atom_dist(rng_);
            s.mol.resize(n);
            for (int i = 0; i < n; ++i) {
                s.mol.pos[i * 3 + 0] = pos_dist(rng_);
                s.mol.pos[i * 3 + 1] = pos_dist(rng_);
                s.mol.pos[i * 3 + 2] = pos_dist(rng_);
                s.mol.atom_types[i] = type_dist(rng_);
            }
            s.pc = derive_pc(s.mol, n_pc_points, noise_std, rng_);
        }
        indices_.resize(n_samples);
        std::iota(indices_.begin(), indices_.end(), 0);
        std::cout << "[SyntheticDataset] Created " << n_samples << " samples\n";
    }

    size_t size() const override { return samples_.size(); }
    Sample& operator[](size_t idx) override { return samples_[indices_[idx]]; }
    void shuffle() override { std::shuffle(indices_.begin(), indices_.end(), rng_); }

private:
    std::vector<Sample> samples_;
    std::vector<int>    indices_;
    std::mt19937 rng_;
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


// ─────────────────────────────────────────────────────────────────────────────
// MolDiffusion model
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

private:
    ModelConfig cfg_;

    std::vector<float*> param;
    std::vector<size_t> param_size;

    // GPU resources
    cudaStream_t   stream_ = 0;
    cublasHandle_t cublas_ = nullptr;

    // Noise schedule
    NoiseSchedule schedule_;

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



// ─────────────────────────────────────────────────────────────────────────────
// AdamW optimizer (decoupled weight decay)
// ─────────────────────────────────────────────────────────────────────────────
class AdamW {
public:
    AdamW(const std::vector<float*>& params,
        const std::vector<size_t>& sizes,
        float lr = 1e-4f, float beta1 = 0.9f, float beta2 = 0.999f,
        float eps = 1e-8f, float weight_decay = 1e-4f);
    ~AdamW();

    void step(float lr_override = -1.f);
    void zero_grad();

    std::vector<float*>& grads() { return grads_; }

private:
    std::vector<float*> params_, m_, v_, grads_;
    std::vector<size_t> sizes_;
    float lr_, beta1_, beta2_, eps_, wd_;
    int   step_;
};

// ─────────────────────────────────────────────────────────────────────────────
// Exponential moving average of model weights
// ─────────────────────────────────────────────────────────────────────────────
class EMA {
public:
    EMA(const std::vector<float*>& params,
        const std::vector<size_t>& sizes,
        float decay = 0.9999f);
    ~EMA();

    void update();
    void copy_to_params();

private:
    std::vector<float*> params_, shadow_;
    std::vector<size_t> sizes_;
    float decay_;
};

// ─────────────────────────────────────────────────────────────────────────────
// LR schedule: linear warmup + cosine decay
// ─────────────────────────────────────────────────────────────────────────────
float cosine_lr(int step, int warmup, int total, float base_lr, float min_lr);

// ─────────────────────────────────────────────────────────────────────────────
// Trainer
// ─────────────────────────────────────────────────────────────────────────────
class Trainer {
public:
    static void train(MolDiffusion& model,
        Dataset& dataset,
        const TrainConfig& cfg);
};

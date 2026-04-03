#ifndef MOL_DIFFUSION_TRAINER_H
#define MOL_DIFFUSION_TRAINER_H

#include <vector>
#include <string>
#include <memory>

#include "mol_diffusion_config.h"
#include "mol_diffusion_model.h"
#include "mol_diffusion_dataset.h"

namespace mol_diffusion {

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

} // namespace mol_diffusion

#endif // MOL_DIFFUSION_TRAINER_H

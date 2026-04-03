#ifndef MOL_DIFFUSION_DATASET_H
#define MOL_DIFFUSION_DATASET_H

#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <filesystem>
#include <memory>
#include <iostream>

#include "mol_diffusion_types.h"
#include "mol_diffusion_utils.h"

// OpenBabel includes (optional - for SDF support)
#ifdef USE_OPENBABEL
#include <openbabel/mol.h>
#include <openbabel/obconversion.h>
#include <openbabel/forcefield.h>
#include <openbabel/atom.h>
#include <openbabel/bond.h>
#include <openbabel/elements.h>
#include <openbabel/obiter.h>
#include <openbabel/generic.h>
using namespace OpenBabel;
#endif

namespace mol_diffusion {

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

} // namespace mol_diffusion

#endif // MOL_DIFFUSION_DATASET_H

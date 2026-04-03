#ifndef MOL_DIFFUSION_UTILS_H
#define MOL_DIFFUSION_UTILS_H

#include <string>
#include <vector>
#include <random>
#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_map>
#include "mol_diffusion_types.h"

namespace mol_diffusion {

// ─────────────────────────────────────────────────────────────────────────────
// Parse helpers
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
// Point cloud derivation from molecule
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

} // namespace mol_diffusion

#endif // MOL_DIFFUSION_UTILS_H

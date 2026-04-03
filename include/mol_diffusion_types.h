#ifndef MOL_DIFFUSION_TYPES_H
#define MOL_DIFFUSION_TYPES_H

#include <vector>
#include <Eigen/Dense>

namespace mol_diffusion {

// ─────────────────────────────────────────────────────────────────────────────
// Atom type mapping (matches Python reference code)
// ─────────────────────────────────────────────────────────────────────────────
constexpr int N_ATOM_TYPES = 11;
static const char* ATOM_SYMBOLS[N_ATOM_TYPES] = {
    "H", "C", "N", "O", "F", "S", "Cl", "Br", "P", "I",  "B"
};
static const int ATOM_VALENCE[N_ATOM_TYPES] = {
    1,   4,   3,   2,   1,   2,    1,    1,   5,   1,   3
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

} // namespace mol_diffusion

#endif // MOL_DIFFUSION_TYPES_H

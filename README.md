# MolDiffusion - C++/CUDA Molecular Diffusion Model

A high-performance implementation of molecular diffusion models for 3D molecule generation, written in C++ and CUDA.

## Project Structure

```
mol_diffusion/
├── include/                    # Public header files
│   ├── mol_diffusion.h         # Main umbrella header
│   ├── mol_diffusion_config.h  # Configuration macros
│   ├── mol_diffusion_types.h   # Core data types (Molecule, PointCloud, Sample)
│   ├── mol_diffusion_utils.h   # Utility functions
│   ├── mol_diffusion_dataset.h # Dataset classes (XYZDataset, SyntheticDataset)
│   ├── mol_diffusion_model.h   # Model definition (MolDiffusion class)
│   └── mol_diffusion_trainer.h # Training utilities (AdamW, EMA, Trainer)
├── src/                        # Implementation files
│   ├── mol_diffusion_model.cu  # CUDA kernels and model implementation
│   ├── mol_diffusion_trainer.cpp # Training loop and optimizer
│   ├── molecule_builder.cpp    # OpenBabel-based molecule processing
│   └── main.cpp                # Entry point with CLI parsing
├── data/                       # Data directory for datasets
├── utils/                      # Utility scripts and tools
├── CMakeLists.txt              # Build configuration
└── README.md                   # This file
```

## Features

- **CUDA-accelerated training** with custom kernels for:
  - EGNN (Equivariant Graph Neural Network) message passing
  - PointNet++ point cloud encoding
  - DDPM diffusion forward/reverse processes
  - Cosine noise schedule
  
- **Multiple dataset formats**:
  - XYZ files
  - SDF/MOL2 files (via OpenBabel)
  - Synthetic data generation

- **Training features**:
  - AdamW optimizer with decoupled weight decay
  - Exponential Moving Average (EMA)
  - Linear warmup + cosine decay learning rate schedule
  - Checkpoint saving/loading

- **Molecule generation**:
  - Conditioned on input point clouds
  - OpenBabel integration for bond perception
  - Force field relaxation (MMFF94/UFF)
  - SMILES output

## Building

### Prerequisites

- CUDA Toolkit 11.0+
- CMake 3.18+
- Eigen3
- OpenBabel 3.0+ (optional, for SDF support)

### Build Steps

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Custom Architecture

For specific GPU architectures:

```bash
cmake .. -DCMAKE_CUDA_ARCHITECTURES="80;86"
```

## Usage

### Training

```bash
# Train on XYZ files
./mol_diffusion --xyz_dir /path/to/xyz/files --epochs 200 --batch_size 32

# Train on SDF files
./mol_diffusion --sdf_dir /path/to/molecules.sdf --epochs 200

# Train with synthetic data
./mol_diffusion --n_samples 50000 --epochs 100
```

### Generation

```bash
# Generate molecules from a checkpoint
./mol_diffusion --ckpt checkpoints/epoch_199.bin \
                --pc_file input_pc.xyz \
                --out_sdf generated.sdf \
                --n_atoms 20 \
                --n_samples 100
```

### Command Line Options

#### Training Options
| Option | Description | Default |
|--------|-------------|---------|
| `--xyz_dir DIR` | Directory of .xyz files | - |
| `--sdf_dir PATH` | SDF file or directory | - |
| `--n_samples N` | Synthetic dataset size | 50000 |
| `--n_pc_points N` | Point cloud size | 256 |
| `--epochs N` | Number of epochs | 200 |
| `--batch_size N` | Batch size | 32 |
| `--lr F` | Learning rate | 1e-4 |
| `--warmup_steps N` | LR warmup steps | 1000 |
| `--feat_dim N` | EGNN feature dimension | 512 |
| `--cond_dim N` | PointNet++ output dim | 1024 |
| `--n_layers N` | Number of EGNN layers | 8 |
| `--ckpt_dir DIR` | Checkpoint directory | checkpoints |
| `--log_dir DIR` | Log directory | runs |

#### Generation Options
| Option | Description | Default |
|--------|-------------|---------|
| `--ckpt PATH` | Model checkpoint (required) | - |
| `--pc_file PATH` | Input point cloud file | random |
| `--out_sdf PATH` | Output SDF file | generated.sdf |
| `--n_atoms N` | Atoms per molecule | 20 |
| `--n_samples N` | Molecules to generate | 4 |
| `--no_relax` | Skip FF relaxation | false |
| `--bond_tol F` | Bond tolerance | 1.15 |

## API Example

```cpp
#include <mol_diffusion.h>

int main() {
    // Configure model
    mol_diffusion::ModelConfig mcfg;
    mcfg.feat_dim = 256;
    mcfg.n_layers = 6;
    
    // Create model
    mol_diffusion::MolDiffusion model(mcfg);
    
    // Create dataset
    mol_diffusion::SyntheticDataset dataset(10000);
    
    // Configure training
    mol_diffusion::TrainConfig tcfg;
    tcfg.epochs = 100;
    tcfg.batch_size = 32;
    
    // Train
    mol_diffusion::Trainer::train(model, dataset, tcfg);
    
    // Save
    model.save("checkpoints/final.bin");
    
    return 0;
}
```

## Architecture

### Model Components

1. **PointNet++ Encoder**: Encodes input point clouds to conditioning vectors
2. **EGNN Denoiser**: Equivariant graph neural network for denoising
3. **Output Heads**: Position and atom type prediction

### Diffusion Process

- **Forward**: Cosine noise schedule with T=1000 steps
- **Reverse**: Learned denoising with EGNN

### Key Hyperparameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `MODEL_FEAT_DIM` | EGNN feature dimension | 512 |
| `MODEL_COND_DIM` | Conditioning dimension | 1024 |
| `MODEL_N_LAYERS` | EGNN layers | 8 |
| `MODEL_RADIUS` | Graph cutoff (Å) | 5.0 |
| `DIFF_T` | Diffusion timesteps | 1000 |

## License

MIT License

## Acknowledgments

This implementation is inspired by:
- EDM (Elucidating the Design Space of Diffusion-Based Generative Models)
- EGNN (E(n) Equivariant Graph Neural Networks)
- PointNet++ for point cloud processing

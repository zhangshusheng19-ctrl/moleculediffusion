#ifndef MOL_DIFFUSION_H
#define MOL_DIFFUSION_H

// =============================================================================
//  MolDiffusion - C++/CUDA Molecular Diffusion Model
// =============================================================================
//  
//  This is the main header file that includes all public API components.
//  The implementation is split across multiple source files:
//    - src/mol_diffusion_model.cu   : Model forward/backward pass
//    - src/mol_diffusion_trainer.cpp: Training loop, optimizer
//    - src/molecule_builder.cpp     : OpenBabel-based molecule processing
//
//  Usage example:
//    #include <mol_diffusion.h>
//    
//    mol_diffusion::ModelConfig mcfg;
//    mol_diffusion::MolDiffusion model(mcfg);
//    
//    mol_diffusion::TrainConfig tcfg;
//    mol_diffusion::SyntheticDataset dataset(10000);
//    
//    mol_diffusion::Trainer::train(model, dataset, tcfg);
//
// =============================================================================

#include "mol_diffusion_config.h"
#include "mol_diffusion_types.h"
#include "mol_diffusion_utils.h"
#include "mol_diffusion_dataset.h"
#include "mol_diffusion_model.h"
#include "mol_diffusion_trainer.h"

// Legacy compatibility: include everything in global namespace too
using namespace mol_diffusion;

#endif // MOL_DIFFUSION_H

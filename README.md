# Conditional-Diffusion-Model-for-Semantically-Aware-3D-Point-Cloud-Generation

This repository contains the implementation for the paper:

**Conditional-Diffusion-Model-for-Semantically-Aware-3D-Point-Cloud-Generation**  
![Conditional-Diffusion Architecture](./README_Assets/architecture.png)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/sushmitaSARKER/Conditional-Diffusion-Model-for-Semantically-Aware-3D-Point-Cloud-Generation.git
   cd Conditional-Diffusion-Model-for-Semantically-Aware-3D-Point-Cloud-Generation
   ```

2. Installing Dependencies

`conda env create -f environment.yaml`

Below is a tested combination of library versions that are compatible. Start here if you are having trouble. 

#### Environment Details
- **Python**: 3.10.4
- **PyTorch**: 2.4.1 (`py3.10_cuda12.1_cudnn9.1.0_0` build)
- **PyTorch-CUDA**: 12.1
- **PyTorch3D**: 0.7.8 (`py310_cu121_pyt241` build)
- **Torch-Geometric**: 2.6.1
- **Torch-Cluster**: 1.6.3 (`+pt24cu121` build)
- **Torch-Scatter**: 2.1.2 (`+pt24cu121` build)
- **Torch-Sparse**: 0.6.18 (`+pt24cu121` build)
- **Torch-Spline-Conv**: 1.2.2 (`+pt24cu121` build)

#### Notes
- Make sure to install the exact versions listed above to avoid compatibility issues.
- These libraries are designed to work with CUDA 12.1 and cuDNN 9.1.0, so ensure your system supports these versions.
- All other libraries should be fairly easy to pip/conda install.


# Training

Download the ShapeNet-Part using their [website](https://shapenet.org/)

# Run through the provided autoencoder training/testing script:

`CDM_Unguided/train_ae.py` will produce model checkpoints located in `CDM_Unguided/logs_ae/<Object>_<idx>_AE_<Date-Time>/checkpoint_X_X.pth`

`CDM_Guided/train_ae.py` will produce model checkpoints located in `CDM_Guided/logs_ae/<Object>_<idx>_AE_<Date-Time>/checkpoint_X_X.pth`

# Run through the provided generation training script:

`CDM_Guided/train_gen.py` will produce model checkpoints located in `CDM_Guided/logs_gen/<Object>_<idx>_AE_<Date-Time>/checkpoint_X_X.pth`

`CDM_Guided/train_gen.py` will produce model checkpoints located in `CDM_Guided/logs_gen/<Object>_<idx>_AE_<Date-Time>/checkpoint_X_X.pth`

For visualizing the snapshots, open up a tensorboard session located in the logs_ae / logs_gen folder. 

These contain numeric metrics & pointcloud visualizations.

# Results

**Comparison of unguided point cloud generation performance for different data splits.**  
JSD, COV, and 1-NNA are multiplied by 10², while MMD is multiplied by 10³.  
The best scores between data splits are highlighted in **bold**.  
(↑: higher is better, ↓: lower is better)

| Category       | JSD ↓ (preset) | JSD ↓ (random) | MMD ↓ (preset) | MMD ↓ (random) | COV ↑ (%) (preset) | COV ↑ (%) (random) | 1-NNA ↓ (preset) | 1-NNA ↓ (random) |
|----------------|----------------|----------------|----------------|----------------|---------------------|---------------------|-------------------|-------------------|
| Bed-1          | 11.24          | **6.10**       | 1.47           | **1.30**       | 54.17               | **82.87**           | 75.00             | **69.11**         |
| Chair-1        | **3.06**       | 3.50           | 1.00           | **0.87**       | **34.10**           | 26.96               | **75.88**         | 76.75             |
| Chair-3        | 32.91          | **31.24**      | 2.34           | **2.33**       | 4.77                | **5.00**            | 99.30             | **98.89**         |
| Dishwasher-3   | **2.87**       | 8.45           | 0.74           | **0.68**       | **89.82**           | 65.71               | **60.53**         | 77.21             |
| Hat-1          | 11.35          | **7.40**       | **0.60**       | 0.81           | **100.00**          | 78.61               | 68.75             | **57.26**         |
| Scissors-1     | 15.65          | **9.85**       | 1.72           | **1.65**       | 91.94               | **94.67**           | **49.98**         | 51.10             |
| TrashCan-1     | 6.79           | **3.54**       | 0.94           | **0.87**       | 70.27               | **75.00**           | 79.43             | **60.94**         |


**Comparison of AutoEncoder reconstruction error**  
(Chamfer Distance multiplied by 10²).  
Algorithms **U** and **G** refer to models trained with a semantically *unguided* and *guided* diffusion process, respectively.  
The numbers **1**, **2**, and **3** correspond to the three levels of ShapeNet-Part segmentation: coarse-, middle-, and fine-grained.  
Dashed lines indicate levels that are not defined.  
The best scores between the unguided and guided models are highlighted in **bold**.


| Model        | Avg ↓ | Bag   | Bed   | Bott  | Bowl   | Chair  | Clock | Dish  | Disp  | Door  | Ear   | Fauc  | Hat   | Key   | Knife | Lamp   | Lap   | Micro  | Mug   | Frid  | Scis  | Stora | Table | Trash | Vase  |
|--------------|--------|-------|-------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| **U1 (Ours)** | 33.59 | 15.39 | 20.42 | 4.52   | 178.02 | 7.54   | 6.42   | 3.72   | 4.24   | 6.08   | **20.41** | 12.32  | 6.80   | 2.80   | **72.45** | 73.12  | 1.94   | 203.13 | **9.21** | 98.10  | 15.99  | 8.32   | 22.59  | 6.10   | 6.60   |
| **U2 (Ours)** | 50.40 | -     | 44.23 | -      | -      | 50.17  | -      | 5.88   | -      | **7.15** | -      | -      | -      | -      | -      | 168.51 | -      | 7.90   | -      | 7.79   | -      | 25.29  | 136.67 | -      | -      |
| **U3 (Ours)** | 92.67 | -     | 79.07 | 6.60   | -      | 90.52  | 10.10  | 7.32   | 4.14   | 8.52   | 38.47  | 18.16  | -      | -      | **17.77** | 642.97 | -      | 8.53   | -      | 10.72  | -      | 35.19  | 571.25 | 15.53  | 10.46  |
| **Avg**       | 47.54 | 15.39 | 47.91 | 5.56   | 178.02 | 49.41  | 8.26   | 5.64   | 4.19   | 7.25   | 29.44  | 15.24  | 6.80   | 2.80   | **45.11** | 294.87 | 1.94   | 73.19  | **9.21** | 38.87  | 15.99  | 22.93  | 243.50 | 10.82  | 8.53   |
| **G1 (Ours)** | **15.54** | **14.83** | **16.16** | **4.09** | **44.90** | **6.62** | **5.24** | **2.84** | **4.14** | **5.73** | 20.56  | **6.48** | **6.16** | **2.33** | 85.31  | **12.55** | **1.84** | **64.16** | 10.98  | **23.59** | **6.89** | **6.06** | **11.51** | **4.69** | **5.18** |
| **G2 (Ours)** | **22.16** | -     | **27.27** | -      | -      | **36.79** | -      | **5.79** | -      | 8.79   | -      | -      | -      | -      | -      | **20.60** | -      | **5.26** | -      | **6.16** | -      | **11.08** | **77.67** | -      | -      |
| **G3 (Ours)** | **20.38** | -     | **49.89** | **4.30** | -      | **43.05** | **8.26** | **6.04** | **3.68** | **6.88** | **33.49** | **7.03** | -      | -      | 23.65  | **21.59** | -      | **6.21** | -      | **6.62** | -      | **11.22** | **101.97** | **6.92** | **5.66** |
| **Avg**       | **19.36** | **14.83** | **31.11** | **4.20** | **44.90** | **28.82** | **6.75** | **4.89** | **3.91** | **7.13** | **27.03** | **6.76** | **6.16** | **2.33** | 54.48  | **18.25** | **1.84** | **25.21** | 10.98  | **12.12** | **6.89** | **9.45** | **63.72** | **5.81** | **5.42** |



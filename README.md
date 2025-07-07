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


# Supplementary Materials


**Detailed comparison of point cloud generation performance for different data splits.**

JSD, COV, and 1-NNA are multiplied by 10², while MMD is multiplied by 10³.
The best scores between data splits are highlighted in bold.
(↑: higher is better, ↓: lower is better)



| Category         | JSD ↓ (preset) | JSD ↓ (random) | MMD ↓ (preset) | MMD ↓ (random) | COV ↑ (%) (preset) | COV ↑ (%) (random) | 1-NNA ↓ (preset) | 1-NNA ↓ (random) |
|------------------|----------------|----------------|----------------|----------------|---------------------|---------------------|-------------------|-------------------|
| Bag-1 | 21.12 | **6.62** | 2.21 | **1.03** | 61.23 | **88.75** | 58.13 | **54.00** |
| Bed-1 | 11.24 | **6.10** | 1.47 | **1.30** | 54.17 | **82.87** | 75.00 | **69.11** |
| Bed-2 | 11.20 | **6.20** | 1.76 | **1.17** | 45.83 | **70.44** | 74.49 | **63.16** |
| Bed-3 | 17.47 | **7.51** | 1.67 | **1.31** | 45.83 | **74.96** | 75.00 | **67.11** |
| Bottle-1 | **4.70** | **4.25** | **0.86** | 0.90 | **66.13** | 53.22 | **66.69** | 73.56 |
| Bottle-3 | 5.89 | **4.80** | **0.88** | 0.90 | **64.86** | 49.43 | **67.57** | 81.61 |
| Bowl-1 | 47.99 | **3.71** | 2.15 | **0.73** | 13.26 | **80.08** | 97.22 | **67.57** |
| Chair-1 | **3.06** | 3.50 | 1.00 | **0.87** | **34.10** | 26.96 | **75.88** | 76.75 |
| Chair-2 | 35.41 | **9.24** | 2.43 | **1.26** | 07.52 | **21.51** | 98.75 | **87.12** |
| Chair-3 | 32.91 | **31.24** | 2.34 | **2.33** | 4.77 | **5.00** | 99.30 | **98.89** |
| Clock-1 | 10.74 | **8.40** | 1.84 | **1.70** | **38.24** | 29.65 | **77.00** | 79.09 |
| Clock-3 | 11.27 | **10.32** | 1.84 | **1.69** | **46.69** | 31.34 | **80.00** | 80.45 |
| Dishwasher-1 | **3.23** | 6.92 | 0.73 | **0.68** | **89.47** | 50.00 | **57.89** | 86.11 |
| Dishwasher-2 | **3.09** | 7.15 | 0.72 | **0.65** | **91.54** | 64.91 | **65.79** | 77.47 |
| Dishwasher-3 | **2.87** | 8.447 | 0.74 | **0.68** | **89.82** | 65.71 | **60.53** | 77.21 |
| Display-1 | 13.64 | **10.83** | 1.45 | **1.13** | **48.89** | 43.28 | **72.47** | 77.57 |
| Display-3 | 15.35 | **12.02** | 1.58 | **1.30** | **45.22** | 41.49 | **70.26** | 72.77 |
| Door-1 | **18.76** | 18.97 | **2.14** | 2.49 | **66.69** | 60.31 | 77.10 | **75.52** |
| Door-2 | 23.38 | **19.52** | 2.43 | **2.10** | **58.76** | 53.71 | **79.24** | 82.04 |
| Door-3 | **19.52** | 24.04 | **1.96** | 2.75 | **64.94** | 52.15 | **76.69** | 77.72 |
| Earphone-1 | **11.09** | 12.95 | 1.85 | **1.20** | **71.76** | 68.89 | **72.42** | 75.56 |
| Earphone-3 | 16.35 | **13.73** | **1.21** | 1.77 | **75.00** | 71.17 | 72.14 | **65.28** |
| Faucet-1 | 12.90 | **11.08** | 1.43 | **1.29** | **79.52** | 62.06 | 71.60 | **69.64** |
| Faucet-3 | 16.55 | **16.43** | 1.49 | **1.40** | **58.02** | 54.63 | 73.87 | **70.27** |
| Hat-1 | 11.35 | **7.40** | **0.60** | 0.81 | **100.00** | 78.61 | 68.75 | **57.26** |
| Keyboard-1 | **17.41** | 19.46 | 1.97 | **1.42** | **37.76** | 35.48 | **76.61** | 78.47 |
| Knife-1 | 11.95 | **10.52** | **1.46** | 1.58 | **72.97** | 64.49 | **62.22** | 67.17 |
| Knife-3 | 11.87 | **10.39** | **1.42** | 1.56 | **56.21** | 53.21 | **60.34** | 68.46 |
| Lamp-1 | **15.97** | 16.96 | **2.11** | 2.19 | **22.65** | 18.00 | **81.20** | 84.88 |
| Lamp-2 | 21.04 | **19.29** | **2.25** | 2.34 | **17.95** | 16.00 | **83.12** | 85.50 |
| Lamp-3 | **21.90** | 26.93 | **2.41** | 3.29 | **20.45** | 11.24 | **81.65** | 90.16 |
| Laptop-1 | 9.76 | **9.09** | 0.77 | **0.43** | **48.12** | 33.72 | 95.56 | **90.70** |
| Microwave-1 | 10.90 | **3.39** | 1.20 | **0.89** | 33.33 | **52.78** | **83.33** | 84.72 |
| Microwave-2 | 8.44 | **2.75** | 1.15 | **0.84** | 33.33 | **69.44** | 79.17 | **74.27** |
| Microwave-3 | 10.80 | **2.69** | 1.22 | **0.85** | 33.33 | **72.22** | 79.17 | **62.50** |
| Mug-1 | 6.40 | **3.76** | 0.90 | **0.66** | 77.63 | **79.41** | **59.49** | 60.53 |
| Refrigerator-1 | 10.08 | **5.80** | 0.71 | **0.67** | 40.00 | **64.86** | 84.22 | **71.62** |
| Refrigerator-2 | 10.45 | **5.24** | 0.72 | **0.67** | 40.00 | **67.57** | 80.21 | **76.65** |
| Refrigerator-3 | 10.14 | **6.30** | 0.68 | **0.65** | 40.00 | **72.16** | 81.81 | **72.97** |
| Scissors-1 | 15.65 | **9.85** | 1.72 | **1.65** | 91.94 | **94.67** | **49.98** | 51.10 |
| Storage Furniture-1 | **4.32** | 5.86 | **1.20** | 1.21 | **32.17** | 22.25 | **85.87** | 89.88 |
| Storage Furniture-2 | **7.27** | 9.61 | **1.19** | 1.23 | **28.70** | 18.75 | **56.87** | 91.75 |
| Storage Furniture-3 | **8.26** | 9.98 | **1.25** | **1.25** | **28.43** | 19.25 | **90.00** | 91.75 |
| Table-1 | 14.79 | **11.84** | **0.97** | 1.39 | 17.25 | **20.19** | **86.28** | 89.12 |
| Table-2 | 40.05 | **29.88** | **1.83** | 2.03 | 7.50 | **10.5** | **92.90** | 93.62 |
| Table-3 | **30.72** | 39.83 | **1.69** | 1.87 | **12.00** | 9.50 | **92.50** | 94.62 |
| TrashCan-1 | 6.79 | **3.54** | 0.94 | **0.87** | 70.27 | **75.00** | 79.43 | **60.94** |
| TrashCan-3 | 6.32 | **4.63** | **0.90** | **0.90** | 72.97 | **77.34** | 87.31 | **65.62** |
| Vase-1 | **4.41** | 4.60 | 1.16 | **1.13** | **62.34** | 40.93 | **72.41** | 73.95 |
| Vase-3 | **4.83** | 4.92 | 1.17 | **1.13** | **51.96** | 42.79 | 75.02 | **72.66** |

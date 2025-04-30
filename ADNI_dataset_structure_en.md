# ADNI Dataset Structure Documentation

## Dataset Overview

The ADNI (Alzheimer's Disease Neuroimaging Initiative) dataset contains brain MRI images for Alzheimer's disease research. This project uses a subset of the ADNI dataset, including three main disease states:

- **AD (Alzheimer's Disease)**: Patients diagnosed with Alzheimer's disease
- **CN (Cognitive Normal)**: Cognitively normal control group
- **MCI (Mild Cognitive Impairment)**: Patients with mild cognitive impairment

## File Format

The original data is stored in NIfTI format (.nii or .nii.gz), which is a common 3D imaging format in neuroimaging. Each NIfTI file contains a complete 3D brain scan.

## Directory Structure

The ADNI_dataset directory is organized according to the following structure:

```
ADNI_dataset/
├── train/             # Training set
│   ├── AD/           # Alzheimer's Disease patients (126 samples)
│   ├── CN/           # Cognitive Normal controls (148 samples)
│   └── MCI/          # Mild Cognitive Impairment patients (263 samples)
├── validation/        # Validation set
│   ├── AD/           # Alzheimer's Disease patients (10 samples)
│   ├── CN/           # Cognitive Normal controls (12 samples)
│   └── MCI/          # Mild Cognitive Impairment patients (22 samples)
└── test/              # Test set
    ├── AD/           # Alzheimer's Disease patients (16 samples)
    ├── CN/           # Cognitive Normal controls (19 samples)
    └── MCI/          # Mild Cognitive Impairment patients (32 samples)
```

## Data Characteristics

1. **Sample Distribution**: In each set, the MCI category has the most samples, followed by CN, with AD having the fewest samples. This imbalanced distribution may affect model training effectiveness.

2. **File Naming Convention**: Filenames follow this format:
   ```
   [Subject ID]_ADNI_[Scan Identifier]_MR_[Scan Parameters].nii
   ```
   Example: `002_S_0619_ADNI_002_S_0619_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20081015124826321_S55372_I120964.nii`

3. **Data Features**:
   - Each NIfTI file contains a 3D brain MRI scan
   - Images typically have different resolutions and dimensions
   - Data requires preprocessing before it can be used for deep learning model training

## Data Preprocessing

To use these 3D NIfTI data for training a Vision Transformer model, the following preprocessing steps are necessary:

1. **Slice Extraction**: Extract 2D slices from 3D scans. Common slice orientations include:
   - Sagittal plane (axis=0)
   - Coronal plane (axis=1)
   - Axial plane (axis=2)

2. **Resizing**: Adjust slices to a uniform size (192x160)

3. **Normalization**: Normalize pixel values

4. **Data Augmentation**: Due to class imbalance, data augmentation might be needed for minority classes (AD, CN)

## Sample Statistics

|  Category  | Training | Validation | Test | Total |
|------------|----------|------------|------|-------|
| AD         | 126      | 10         | 16   | 152   |
| CN         | 148      | 12         | 19   | 179   |
| MCI        | 263      | 22         | 32   | 317   |
| **Total**  | 537      | 44         | 67   | 648   | 
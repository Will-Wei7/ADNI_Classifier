# ADNI Dataset PNG Structure

This document describes the structure of the `ADNI_dataset_png_axial` directory, which contains PNG slices converted from ADNI NIfTI files using the `3_nii_to_png_all.py` script.

## Directory Structure

```
ADNI_dataset_png_axial/
├── train/
│   ├── AD/                   # Alzheimer's Disease subjects
│   │   ├── 002_S_0413/       # Patient ID folder
│   │   │   ├── 002_S_0413_slice_000.png
│   │   │   ├── 002_S_0413_slice_001.png
│   │   │   └── ...
│   │   ├── 003_S_1059/
│   │   └── ...
│   ├── CN/                   # Cognitively Normal subjects
│   │   ├── 002_S_0559/
│   │   ├── 005_S_0553/
│   │   └── ...
│   └── MCI/                  # Mild Cognitive Impairment subjects
│       ├── 002_S_1018/
│       ├── 003_S_4288/
│       └── ...
├── validation/
│   ├── AD/
│   ├── CN/
│   └── MCI/
└── test/
    ├── AD/
    ├── CN/
    └── MCI/
```

## Naming Convention

Each PNG file in the dataset follows this naming convention:

```
[patient_ID]_slice_[slice_number].png
```

Where:
- `patient_ID` is a short identifier for the patient (e.g., "002_S_0413")
- `slice_number` is a zero-padded three-digit number representing the slice index (e.g., "000", "001", "002")

## File Format Details

- **File Format**: PNG (Portable Network Graphics)
- **Color Mode**: Grayscale (8-bit)
- **Normalization**: Each slice is normalized to the 0-255 range for optimal contrast
- **Orientation**: Axial slices (top-to-bottom view of brain), rotated for proper anatomical orientation

## Additional Information

The PNG files were generated from the original ADNI NIfTI files using the following processing steps:

1. Loading the 3D NIfTI volume
2. Extracting axial (horizontal) slices
3. Normalizing pixel values to the 0-255 range
4. Rotating images for proper anatomical orientation
5. Converting to 8-bit grayscale PNG format

This structure preserves the original train/validation/test split and diagnostic categories from the ADNI dataset while providing a more accessible image format for image-based deep learning models.

## Usage Notes

- Each folder represents a single subject's brain MRI
- Slices are numbered sequentially from superior (top of head) to inferior (bottom of head)
- The diagnostic labels (AD/CN/MCI) are preserved from the original dataset structure 
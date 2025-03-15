# DICOM to JPG Conversion for Alzheimer's Prediction

This repository contains scripts to convert DICOM medical images to JPG format and use them for Alzheimer's disease prediction using a pre-trained Vision Transformer (ViT) model.

## Requirements

Install the required packages:

```bash
pip install pydicom numpy pillow matplotlib torch torchvision
```

## Scripts

### 1. DICOM to JPG Converter (`convert_dicom_to_jpg.py`)

This script converts DICOM files to JPG format, which can then be used for prediction.

#### Usage

**Convert a single DICOM file:**

```bash
python convert_dicom_to_jpg.py path/to/file.dcm
```

**Convert a single DICOM file with custom output path:**

```bash
python convert_dicom_to_jpg.py path/to/file.dcm --output path/to/output.jpg
```

**Convert all DICOM files in a directory:**

```bash
python convert_dicom_to_jpg.py path/to/dicom_directory --output path/to/output_directory
```

**Convert all DICOM files in a directory and subdirectories:**

```bash
python convert_dicom_to_jpg.py path/to/dicom_directory --output path/to/output_directory --recursive
```

**Display a comparison of the original DICOM and converted JPG:**

```bash
python convert_dicom_to_jpg.py path/to/file.dcm --display
```

**Customize windowing parameters:**

```bash
python convert_dicom_to_jpg.py path/to/file.dcm --window-center 40 --window-width 80
```

### 2. DICOM to Prediction Pipeline (`dicom_to_prediction.py`)

This script converts a DICOM file to JPG and then runs the Alzheimer's prediction model on it.

#### Usage

**Convert and predict:**

```bash
python dicom_to_prediction.py path/to/file.dcm
```

**Convert with custom output path and predict:**

```bash
python dicom_to_prediction.py path/to/file.dcm --output path/to/output.jpg
```

**Convert only (skip prediction):**

```bash
python dicom_to_prediction.py path/to/file.dcm --no-predict
```

### 3. Alzheimer's Prediction Model (`predict_alzheimer.py`)

This script loads a pre-trained Vision Transformer model and predicts the Alzheimer's class of a JPG image.

#### Usage

```bash
python predict_alzheimer.py path/to/image.jpg
```

## Alzheimer's Classes

The model predicts one of four classes:

1. **AD (Alzheimer's Disease)**: Full Alzheimer's disease with significant cognitive decline
2. **LMCI (Late Mild Cognitive Impairment)**: A more advanced stage of cognitive impairment that often precedes Alzheimer's
3. **EMCI (Early Mild Cognitive Impairment)**: An early stage of cognitive impairment
4. **CN (Cognitively Normal)**: No signs of cognitive impairment

## Notes on DICOM Windowing

DICOM images often have a wide dynamic range that needs to be "windowed" to be properly visualized. The windowing parameters (center and width) determine which part of the dynamic range is displayed:

- **Window Center**: The center of the window, representing the midpoint of the range of pixel values to be displayed.
- **Window Width**: The width of the window, representing the range of pixel values to be displayed.

If these parameters are not specified, the script will try to use the values from the DICOM file or calculate reasonable defaults.

## Example Workflow

1. Convert a DICOM brain MRI to JPG:
   ```bash
   python convert_dicom_to_jpg.py brain_mri.dcm --output brain_mri.jpg
   ```

2. Run the prediction on the converted image:
   ```bash
   python predict_alzheimer.py brain_mri.jpg
   ```

3. Or do both in one step:
   ```bash
   python dicom_to_prediction.py brain_mri.dcm
   ``` 
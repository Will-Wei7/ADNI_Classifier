import os
import sys
import argparse
import numpy as np
import pydicom
from PIL import Image
import matplotlib.pyplot as plt

def convert_dicom_to_jpg(dicom_path, output_path=None, window_center=None, window_width=None):
    """
    Convert a DICOM file to JPG format
    
    Parameters:
    -----------
    dicom_path : str
        Path to the DICOM file
    output_path : str, optional
        Path to save the JPG file. If None, will use the same name as the DICOM file with .jpg extension
    window_center : int, optional
        Window center for windowing. If None, will use the value from the DICOM file or a default value
    window_width : int, optional
        Window width for windowing. If None, will use the value from the DICOM file or a default value
    
    Returns:
    --------
    output_path : str
        Path to the saved JPG file
    """
    # Check if the DICOM file exists
    if not os.path.exists(dicom_path):
        raise FileNotFoundError(f"DICOM file {dicom_path} not found")
    
    # Read the DICOM file
    try:
        dicom = pydicom.dcmread(dicom_path)
    except Exception as e:
        raise Exception(f"Error reading DICOM file: {e}")
    
    # Extract the pixel array
    try:
        pixel_array = dicom.pixel_array
    except Exception as e:
        raise Exception(f"Error extracting pixel array: {e}")
    
    # Apply windowing if specified
    if hasattr(dicom, 'WindowCenter') and hasattr(dicom, 'WindowWidth'):
        if window_center is None:
            window_center = dicom.WindowCenter
            if isinstance(window_center, pydicom.multival.MultiValue):
                window_center = window_center[0]
        if window_width is None:
            window_width = dicom.WindowWidth
            if isinstance(window_width, pydicom.multival.MultiValue):
                window_width = window_width[0]
    else:
        # Default windowing values if not in DICOM
        if window_center is None:
            window_center = np.mean(pixel_array)
        if window_width is None:
            window_width = np.max(pixel_array) - np.min(pixel_array)
    
    # Apply windowing
    min_value = window_center - window_width // 2
    max_value = window_center + window_width // 2
    pixel_array = np.clip(pixel_array, min_value, max_value)
    
    # Normalize to 0-255
    pixel_array = ((pixel_array - min_value) / (max_value - min_value)) * 255.0
    pixel_array = pixel_array.astype(np.uint8)
    
    # Create the output path if not specified
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(dicom_path))[0]
        output_path = f"{base_name}.jpg"
    
    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save as JPG
    image = Image.fromarray(pixel_array)
    image.save(output_path)
    
    print(f"Converted {dicom_path} to {output_path}")
    return output_path

def convert_directory(input_dir, output_dir=None, recursive=False, window_center=None, window_width=None):
    """
    Convert all DICOM files in a directory to JPG format
    
    Parameters:
    -----------
    input_dir : str
        Path to the directory containing DICOM files
    output_dir : str, optional
        Path to save the JPG files. If None, will use the same directory as the DICOM files
    recursive : bool, optional
        Whether to recursively search for DICOM files in subdirectories
    window_center : int, optional
        Window center for windowing. If None, will use the value from the DICOM file or a default value
    window_width : int, optional
        Window width for windowing. If None, will use the value from the DICOM file or a default value
    
    Returns:
    --------
    converted_files : list
        List of paths to the saved JPG files
    """
    # Check if the input directory exists
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory {input_dir} not found")
    
    # Create the output directory if it doesn't exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    converted_files = []
    
    # Walk through the directory
    if recursive:
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith('.dcm'):
                    dicom_path = os.path.join(root, file)
                    
                    # Create relative path for output
                    if output_dir:
                        rel_path = os.path.relpath(root, input_dir)
                        out_dir = os.path.join(output_dir, rel_path)
                        if not os.path.exists(out_dir):
                            os.makedirs(out_dir)
                        
                        base_name = os.path.splitext(file)[0]
                        out_path = os.path.join(out_dir, f"{base_name}.jpg")
                    else:
                        base_name = os.path.splitext(file)[0]
                        out_path = os.path.join(root, f"{base_name}.jpg")
                    
                    try:
                        converted_file = convert_dicom_to_jpg(dicom_path, out_path, window_center, window_width)
                        converted_files.append(converted_file)
                    except Exception as e:
                        print(f"Error converting {dicom_path}: {e}")
    else:
        for file in os.listdir(input_dir):
            if file.lower().endswith('.dcm'):
                dicom_path = os.path.join(input_dir, file)
                
                if output_dir:
                    base_name = os.path.splitext(file)[0]
                    out_path = os.path.join(output_dir, f"{base_name}.jpg")
                else:
                    base_name = os.path.splitext(file)[0]
                    out_path = os.path.join(input_dir, f"{base_name}.jpg")
                
                try:
                    converted_file = convert_dicom_to_jpg(dicom_path, out_path, window_center, window_width)
                    converted_files.append(converted_file)
                except Exception as e:
                    print(f"Error converting {dicom_path}: {e}")
    
    return converted_files

def display_comparison(dicom_path, jpg_path):
    """
    Display a comparison of the original DICOM and the converted JPG
    
    Parameters:
    -----------
    dicom_path : str
        Path to the DICOM file
    jpg_path : str
        Path to the JPG file
    """
    # Read the DICOM file
    dicom = pydicom.dcmread(dicom_path)
    dicom_array = dicom.pixel_array
    
    # Read the JPG file
    jpg_image = Image.open(jpg_path)
    jpg_array = np.array(jpg_image)
    
    # Display the images
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(dicom_array, cmap='gray')
    plt.title('Original DICOM')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(jpg_array, cmap='gray')
    plt.title('Converted JPG')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Convert DICOM files to JPG format')
    parser.add_argument('input', help='Path to the DICOM file or directory')
    parser.add_argument('--output', '-o', help='Path to save the JPG file or directory')
    parser.add_argument('--recursive', '-r', action='store_true', help='Recursively search for DICOM files in subdirectories')
    parser.add_argument('--window-center', '-c', type=int, help='Window center for windowing')
    parser.add_argument('--window-width', '-w', type=int, help='Window width for windowing')
    parser.add_argument('--display', '-d', action='store_true', help='Display a comparison of the original DICOM and the converted JPG')
    
    args = parser.parse_args()
    
    try:
        if os.path.isdir(args.input):
            converted_files = convert_directory(
                args.input, 
                args.output, 
                args.recursive, 
                args.window_center, 
                args.window_width
            )
            print(f"Converted {len(converted_files)} DICOM files to JPG")
            
            # Display the first converted file if requested
            if args.display and converted_files:
                dicom_path = os.path.splitext(converted_files[0])[0] + '.dcm'
                display_comparison(dicom_path, converted_files[0])
        else:
            jpg_path = convert_dicom_to_jpg(
                args.input, 
                args.output, 
                args.window_center, 
                args.window_width
            )
            
            # Display the converted file if requested
            if args.display:
                display_comparison(args.input, jpg_path)
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
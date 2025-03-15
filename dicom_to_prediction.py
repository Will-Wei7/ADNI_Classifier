import os
import sys
import argparse
from convert_dicom_to_jpg import convert_dicom_to_jpg
import subprocess

def main():
    parser = argparse.ArgumentParser(description='Convert DICOM file to JPG and predict Alzheimer\'s class')
    parser.add_argument('input', help='Path to the DICOM file')
    parser.add_argument('--output', '-o', help='Path to save the JPG file')
    parser.add_argument('--window-center', '-c', type=int, help='Window center for windowing')
    parser.add_argument('--window-width', '-w', type=int, help='Window width for windowing')
    parser.add_argument('--no-predict', action='store_true', help='Skip prediction step')
    
    args = parser.parse_args()
    
    try:
        # Convert DICOM to JPG
        print("Converting DICOM to JPG...")
        jpg_path = convert_dicom_to_jpg(
            args.input, 
            args.output, 
            args.window_center, 
            args.window_width
        )
        
        # Run prediction if not skipped
        if not args.no_predict:
            print("\nRunning Alzheimer's prediction...")
            predict_cmd = ["python", "predict_alzheimer.py", jpg_path]
            subprocess.run(predict_cmd)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
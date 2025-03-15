#!/bin/bash

# Directory containing the Nondemented images
TEST_DIR="Test_data/Alzheimer_s Dataset/test/NonDemented"

# Output directory for results
RESULTS_DIR="results/nondemented_test"

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# Run batch prediction with visualization
python batch_predict.py "$TEST_DIR" \
    --expected-class "CN" \
    --output-csv "$RESULTS_DIR/nondemented_results.csv" \
    --output-dir "$RESULTS_DIR" \
    --visualize

echo "Testing complete. Results saved to $RESULTS_DIR"
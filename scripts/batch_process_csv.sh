#!/bin/bash

# Batch process all CSV files in the LAFAN1 dataset directory
CSV_DIR="/home/linji/nfs/LAFAN1_Retargeting_Dataset/g1"
SCRIPT_DIR="/home/linji/nfs/whole_body_tracking/scripts"

# Check if CSV directory exists
if [ ! -d "$CSV_DIR" ]; then
    echo "Error: CSV directory $CSV_DIR does not exist"
    exit 1
fi

# Activate conda environment and set required environment variables
source ~/miniconda3/etc/profile.d/conda.sh
conda activate isaac_lab_0817
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Verify environment setup
PYTHON_EXE=$(which python)
echo "Using Python: $PYTHON_EXE"
echo "Python version: $(python --version)"
echo "Environment: $CONDA_DEFAULT_ENV"

# Change to the parent directory so scripts/csv_to_npz.py path works
cd "/home/linji/nfs/whole_body_tracking"

# Process each CSV file
for csv_file in "$CSV_DIR"/*.csv; do
    if [ -f "$csv_file" ]; then
        # Extract filename without path and extension
        filename=$(basename "$csv_file" .csv)
        echo "Processing: $filename"
        
        # Run the csv_to_npz script
        python scripts/csv_to_npz.py \
            --input_file "$csv_file" \
            --input_fps 30 \
            --output_name "$filename" \
            --output_fps 50 \
            --headless
        
        # Check if the process completed successfully
        if [ $? -eq 0 ]; then
            echo "✅ Successfully processed: $filename"
        else
            echo "❌ Failed to process: $filename"
        fi
        
        echo "Completed: $filename"
        echo "---"
    fi
done

echo "All CSV files processed!"
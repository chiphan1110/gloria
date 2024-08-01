#!/bin/bash

# Change to the directory where the Python file is located
cd /data/ssd_2/workenv/Chi/gloria

echo "Preprocess the datasets..."

python /data/ssd_2/workenv/Chi/gloria/gloria/datasets/preprocess_datasets.py \
    --dataset chexpert 

echo "Done."
#!/bin/bash
module load anaconda3/personal
source activate h_impact_env
python setup.py build_ext --inplace
conda deactivate


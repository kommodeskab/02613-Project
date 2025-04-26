#!/bin/sh

# må IKKE indeholde mellemrum
#BSUB -J q10

#BSUB -q gpuv100
#BSUB -R "select[gpu32gb]"

# number of GPUs to use
#BSUB -gpu "num=1:mode=exclusive_process"

# number of cores to use
#BSUB -n 4

# gb memory per core
#BSUB -R "rusage[mem=4GB]"
# cores is on the same slot
#BSUB -R "span[hosts=1]"

# walltime
#BSUB -W 10:00
#BSUB -o hpc/output_%J.out 
#BSUB -e hpc/error_%J.err   

nvidia-smi
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613
python3 Task\ 9-10/q10.py all
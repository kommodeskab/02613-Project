#!/bin/sh

# må IKKE indeholde mellemrum
#BSUB -J q9

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
#BSUB -W 00:10
#BSUB -o hpc/output_%J.out 
#BSUB -e hpc/error_%J.err   

nvidia-smi
module load python3/3.10.16 cuda/12.0
source .venv/bin/activate
python3 Task\ 9-10/q9.py 50
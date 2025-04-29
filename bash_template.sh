#!/bin/sh

# må IKKE indeholde mellemrum
#BSUB -J <job_name>

#BSUB -q gpuv100

# number of GPUs to use
#BSUB -gpu "num=1:mode=exclusive_process"

# number of cores to use
#BSUB -n 4

# gb memory per core
#BSUB -R "rusage[mem=4GB]"
# cores is on the same slot
#BSUB -R "span[hosts=1]"

# walltime
#BSUB -W 1:00
#BSUB -o hpc/output_%J.out 
#BSUB -e hpc/error_%J.err   

# what to do
# (det er en god ide at køre nedenstående i terminalen og se om det virker inden man sætter det i kø)
nvidia-smi
module load python3/3.10.16 cuda/12.0
source .venv/bin/activate
python3 <your_script.py> <your_arguments>
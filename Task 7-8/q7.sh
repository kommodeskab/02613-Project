#!/bin/sh

# må IKKE indeholde mellemrum
#BSUB -J q7


#BSUB -R "select[model==XeonGold6226R]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "span[hosts=1]"

# walltime
#BSUB -W 00:15
#BSUB -o output_%J.out 
#BSUB -e error_%J.err   

module load python3/3.10.16 cuda/12.0
source .venv/bin/activate
python3 Task\ 7-8/q7.py 64
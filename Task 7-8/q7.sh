#!/bin/bash
#BSUB -J qt                          
#BSUB -q hpc                         
#BSUB -W 2:00                        
#BSUB -R "rusage[mem=512MB]"         
#BSUB –n 8
#BSUB –R "span[hosts=1]"sxm2sh
#BSUB -o q7_%J.out                  
#BSUB -e q7_%J.err                  

cd /zhome/7e/7/168289/Desktop/02613-Project/Task 7-8
# Load Python if needed
module load python3

# Run your Python script
python q7.py

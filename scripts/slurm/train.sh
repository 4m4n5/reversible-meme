#!/bin/bash

#SBATCH --job-name="true_no_glove"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=as3ek@virginia.edu
#
#SBATCH --error="/u/as3ek/github/reversible-meme/models/run4/train.err"
#SBATCH --output="/u/as3ek/github/reversible-meme/models/run4/train.out"
#SBATCH --gres=gpu:4

module load cuda-toolkit-10.1
module load cudnn-7.3.1
module load anaconda3

python -u /u/as3ek/github/reversible-meme/train.py --id run4 --use-glove False --batch-size 32

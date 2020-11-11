#!/bin/bash

#SBATCH --job-name="crawler_RMG"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=as3ek@virginia.edu
#
#SBATCH --error="/u/as3ek/github/reversible-meme/scripts/logs/crawler.err"
#SBATCH --output="/u/as3ek/github/reversible-meme/scripts/logs/crawler.output"

module load anaconda3

python -u /u/as3ek/github/reversible-meme/scripts/meme_crawler.py


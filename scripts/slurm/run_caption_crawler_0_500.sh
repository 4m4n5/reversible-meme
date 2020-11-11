#!/bin/bash

#SBATCH --job-name="cp0_500"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=as3ek@virginia.edu
#
#SBATCH --error="/u/as3ek/github/reversible-meme/scripts/logs/crawler_0_500.err"
#SBATCH --output="/u/as3ek/github/reversible-meme/scripts/logs/crawler_0_500.output"

module load anaconda3

python -u /u/as3ek/github/reversible-meme/scripts/caption_crawler.py --sm 0 --em 500


#!/bin/bash

#SBATCH --job-name="cp1000_1400"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=as3ek@virginia.edu
#
#SBATCH --error="/u/as3ek/github/reversible-meme/scripts/logs/crawler_1000_1400.err"
#SBATCH --output="/u/as3ek/github/reversible-meme/scripts/logs/crawler_1000_1400.output"

module load anaconda3

python -u /u/as3ek/github/reversible-meme/scripts/caption_crawler.py --sm 1000 --em 1400

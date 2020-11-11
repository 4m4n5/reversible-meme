#!/bin/bash

#SBATCH --job-name="cp500_1000"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=as3ek@virginia.edu
#
#SBATCH --error="/u/as3ek/github/reversible-meme/scripts/logs/crawler_500_1000.err"
#SBATCH --output="/u/as3ek/github/reversible-meme/scripts/logs/crawler_500_1000.output"

module load anaconda3

python -u /u/as3ek/github/reversible-meme/scripts/caption_crawler.py --sm 500 --em 1000


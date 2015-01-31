#!/bin/bash

#SBATCH --account=pi-yaliamit
#SBATCH --ntasks=1
#SBATCH --job-name=exParts_100_10Parts_seed_1
#SBATCH --output=exParts_100_10Parts_seed_1.out
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=32000
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rcc.jiajun.uchicago@gmail.com

module load python/2.7-2014q2
module load parallel

python trainExtensionPartsModel.py plainParts_100_parts_seed_1.npy 500 5 12 /home/jiajun/.mnist/mnistTrainingData.npy extensionParts_100_10_parts_seed_1.npy 1


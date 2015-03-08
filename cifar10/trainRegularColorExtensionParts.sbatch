#!/bin/bash

#SBATCH --account=pi-yaliamit
#SBATCH --ntasks=1
#SBATCH --job-name=trainRegularColorExtensionParts_cifar
#SBATCH --output=trainRegularColorExtensionParts.out
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=32000
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rcc.jiajun.uchicago@gmail.com

module load python/2.7-2014q2
python trainRegularColorExtensionParts.py trainRegularColorExtensionParts.npy 0 && python test.py trainRegularColorExtensionParts.npy

#!/bin/bash

#SBATCH --account=pi-yaliamit
#SBATCH --ntasks=1
#SBATCH --job-name=train_all_plain_parts
#SBATCH --output=train_all_plain_parts.out
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=32000
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rcc.jiajun.uchicago@gmail.com

module load python/2.7-2014q2
module load parallel

parallel "python trainPlainModel.py {1} 6 /home/jiajun/.mnist/mnistTrainingData.npy plainParts_{1}_parts_seed_{2}.npy {2}" ::: 100 200 ::: {1..10}

#parallel "python trainPlainModel.py {1} 6 /home/jiajun/.mnist/mnistTrainingData.npy plainParts_{1}_parts_seed_{2}.npy {2}" ::: 200 400 ::: {1..10}

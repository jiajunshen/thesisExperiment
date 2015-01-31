#!/bin/bash

module load parallel

parallel "

echo \"#!/bin/bash

#SBATCH --account=pi-yaliamit
#SBATCH --ntasks=1
#SBATCH --job-name=plainParts_{2}_Parts_seed_{1}
#SBATCH --output=plainParts_{2}_Parts_seed_{1}.out
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=32000
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rcc.jiajun.uchicago@gmail.com

module load python/2.7-2014q2
module load parallel

python trainPlainModel.py {2} 6 /home/jiajun/.mnist/mnistTrainingData.npy plainParts_{2}_parts_seed_{1}.npy {1}
\" >> bashTrain_part{2}_seed_{1}.sbatch && sbatch bashTrain_part{2}_seed_{1}.sbatch" ::: {1..10} ::: 100 200

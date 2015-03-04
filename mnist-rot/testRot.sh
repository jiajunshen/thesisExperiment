#!/bin/bash

module load parallel

parallel "

echo \"#!/bin/bash

#SBATCH --account=pi-yaliamit
#SBATCH --ntasks=1
#SBATCH --job-name=MNIST_ROT_rotParts_{2}_Parts_{3}_orientation_{4}_spreading_seed_{1}
#SBATCH --output=/project/yaliamit/jiajun-master/MNIST_ROT/rotPartsModelOutputFile/rotParts_{2}_Parts_{3}_orientation_{4}_spreading_seed_{1}.out
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=32000
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rcc.jiajun.uchicago@gmail.com

module load python/2.7-2014q2
module load parallel

python trainRotModel.py {2} {3} {4} 6 $ROT_MNIST/mnistROT.npy /project/yaliamit/jiajun-master/MNIST_ROT/rotPartsModel/rotParts_{2}_parts_{3}_orientation_{4}_spreading_seed_{1}.npy {1}
\" >> bashTrain_part{2}_orientation_{3}_spreading_{4}_seed_{1}.sbatch && sbatch bashTrain_part{2}_orientation_{3}_spreading_{4}_seed_{1}.sbatch" ::: {1..10} ::: 60 ::: 8 16 ::: 0 1

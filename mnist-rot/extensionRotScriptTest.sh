#!/bin/bash

module load parallel

parallel "

echo \"#!/bin/bash

#SBATCH --account=pi-yaliamit
#SBATCH --ntasks=1
#SBATCH --job-name=rotExtensionParts_{2}_Parts_{3}_extensionParts_{5}_rotations_{4}_shape_seed_{1}
#SBATCH --output=/project/yaliamit/jiajun-master/MNIST-ROT/rotExtensionPartsModelOutputFile/rotExtensionParts_{2}_Parts_{3}_extensionParts_{5}_rotations_{4}_shape_{6}_spreading_seed_{1}.out
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=32000
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rcc.jiajun.uchicago@gmail.com

module load python/2.7-2014q2
module load parallel

python trainRotExtensionPartsModel.py /project/yaliamit/jiajun-master/MNIST-ROT/rotPartsModel/rotParts_{2}_parts_{5}_orientation_{6}_spreading_seed_{1}.npy {2} {3} {5} {4} /home/jiajun/.mnist/mnistTrainingData.npy /project/yaliamit/jiajun-master/MNIST-ROT/rotExtensionPartsModel/rotExtensionParts_{2}_{3}_parts_{5}_rotations_{4}_shape_{6}_spreading_seed_{1}.npy {1}
\" >> bashExtensionPartsTrain_{2}_{3}_parts_{5}_rotation_{6}_spreading_{4}_shape_seed_{1}.sbatch && sbatch bashExtensionPartsTrain_{2}_{3}_parts_{5}_rotation_{6}_spreading_{4}_shape_seed_{1}.sbatch" ::: 1 ::: 25 ::: 5 ::: 10 ::: 2 ::: 0

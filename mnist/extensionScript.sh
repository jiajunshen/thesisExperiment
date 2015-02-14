#!/bin/bash

module load parallel

parallel "

echo \"#!/bin/bash

#SBATCH --account=pi-yaliamit
#SBATCH --ntasks=1
#SBATCH --job-name=extensionParts_{2}_Parts_{3}_extensionParts_{4}_shape_seed_{1}
#SBATCH --output=./extensionPartsModelOutputFile/extensionParts_{2}_Parts_{3}_extensionParts_{4}_shape_seed_{1}.out
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=32000
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rcc.jiajun.uchicago@gmail.com

module load python/2.7-2014q2
module load parallel

python trainExtensionPartsModel.py ./plainPartsModel/plainParts_{2}_parts_seed_{1}.npy {2} {3} {4} /home/jiajun/.mnist/mnistTrainingData.npy ./extensionPartsModel/extensionParts_{2}_{3}_parts_{4}_shape_seed_{1}.npy {1}
\" >> bashExtensionPartsTrain_{2}_{3}_parts_{4}_shape_seed_{1}.sbatch && sbatch bashExtensionPartsTrain_{2}_{3}_parts_{4}_shape_seed_{1}.sbatch" ::: {1..10} ::: 500 ::: 5 10 ::: 10 12 

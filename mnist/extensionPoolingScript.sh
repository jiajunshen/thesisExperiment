#!/bin/bash

module load parallel

parallel "

echo \"#!/bin/bash

#SBATCH --account=pi-evtimov
#SBATCH --ntasks=1
#SBATCH --job-name=extensionPooling_{2}_Parts_{3}_extensionParts_{4}_shape_pooling_{5}_distance_{6}_seed_{1}_new_seed{7}_epoch{8}_nhidden{9}_rate{10}
#SBATCH --output=/project/yaliamit/jiajun-master/extensionPoolingModelOutputFile/extensionPooling_{2}_Parts_{3}_extensionParts_{4}_shape_pooling_{5}_distance_{6}_seed_{1}_new_seed_{7}_epoch{8}_nhidden{9}_rate{10}.out
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=32000
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rcc.jiajun.uchicago@gmail.com

module load python/2.7-2014q2
module load parallel

python trainExtensionPoolingModel.py ./extensionPartsModel/extensionParts_{2}_{3}_parts_{4}_shape_seed_{1}.npy {2} {3} {4} {5} /home/jiajun/.mnist/mnistTrainingData.npy /project/yaliamit/jiajun-master/extensionPoolingModel/extensionPooling_{2}_parts_{3}_extensionParts_{4}_shape_pooling_{5}_distance_{6}_new_seed_{7}_epoch_{8}_nhidden_{9}_rate_{10}_seed{1}.npy /project/yaliamit/jiajun-master/newWeights/extensionPoolingWeights_{2}_parts_{3}_extensionParts_{4}_shape_pooling_{5}_seed_1_new_seed_{7}_epoch_{8}_nhidden_{9}_rate_{10}.npy {6} {1} {7} {8} {9} {10}\" >> bashExtensionPooling_{2}_{3}_parts_{4}_shape_pooling_{5}_distance_{6}_seed_{1}_new_seed_{7}_epoch{8}_nhidden{9}_rate{10}.sbatch && sbatch bashExtensionPooling_{2}_{3}_parts_{4}_shape_pooling_{5}_distance_{6}_seed_{1}_new_seed_{7}_epoch{8}_nhidden{9}_rate{10}.sbatch" ::: 1 ::: 100 ::: 10 ::: 12 ::: 8 ::: 5 ::: {1..5} ::: 100 200 ::: 200 400 500 ::: 10

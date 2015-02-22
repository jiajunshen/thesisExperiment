#!/bin/bash

module load parallel

parallel "

echo \"#!/bin/bash

#SBATCH --account=pi-evtimov
#SBATCH --ntasks=1
#SBATCH --job-name=extensionPooling_{2}_Parts_{3}_extensionParts_{4}_shape_pooling_{5}_distance_{6}_seed_{1}
#SBATCH --output=/project/yaliamit/jiajun-master/extensionPoolingModelOutputFile/extensionPooling_{2}_Parts_{3}_extensionParts_{4}_shape_pooling_{5}_distance_{6}_seed_{1}.out
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=32000
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rcc.jiajun.uchicago@gmail.com

module load python/2.7-2014q2
module load parallel


######UNFINISHED#####
python trainExtensionPoolingModel.py ./extensionPartsModel/extensionParts_{2}_{3}_parts_{4}_shape_seed_{1}.npy {2} {3} {4} {5} /mnt/research_disk_1/newhome/jiajun/.mnist/mnistTrainingData.npy /mnt/research_disk_1/newhome/jiajun/Documents/extensionPoolingModel/extensionPooling_{2}_parts_{3}_extensionParts_{4}_shape_pooling_{5}_distance_{6}_seed_{1}.npy /project/yaliamit/jiajun-master/extensionPoolingWeightsBackUp/extensionPoolingWeights_{2}_parts_{3}_extensionParts_{4}_shape_pooling_{5}_seed_1.npy {6} {1}\" >> bashExtensionPooling_{2}_{3}_parts_{4}_shape_pooling_{5}_distance_{6}_seed_{1}.sbatch && sbatch bashExtensionPooling_{2}_{3}_parts_{4}_shape_pooling_{5}_distance_{6}_seed_{1}.sbatch" ::: {5..10} ::: 100 ::: 5 10 ::: 10 12 ::: 4 8 ::: 5 10

#!/bin/bash

module load parallel

parallel "

echo \"#!/bin/bash

#SBATCH --account=pi-yaliamit
#SBATCH --ntasks=1
#SBATCH --job-name=extensioClassification_{2}_parts_{3}_extensionParts_{4}_shape_{5}_pooling_{6}_classifier_{7}_numOfClass_{8}_trainingSize_{9}_classificationType
#SBATCH --output=/project/yaliamit/jiajun-master/MNIST_ROT/extensionClassificationModelOutputFile/extensionPoolingClassification_{2}_parts_{3}_extensionParts_{4}_shape_{5}_pooling_{6}_classifier_{7}_numOfClass_{8}_trainingSize_{9}_classificationType.out
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=32000
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rcc.jiajun.uchicago@gmail.com

module load python/2.7-2014q2
module load parallel

python trainClassification.py /project/yaliamit/jiajun-master/MNIST_ROT/extensionPartsModel/extensionParts_{2}_{3}_parts_{4}_shape_seed_ $ROT_MNIST/mnistROT.npy $ROT_MNIST/mnistROTLabel.npy /project/yaliamit/jiajun-master/MNIST_ROT/extensionClassificationModel/extensionClassification_{2}_parts_{3}_extensionParts_{4}_shape_{5}_pooling_{6}_classifier_{7}_numOfClass_{8}_trainingSize_{9}_classificationType_seed_ $ROT_MNIST/mnistROTTEST.npy $ROT_MNIST/mnistROTLABELTEST.npy {1} {6} {7} {10} ./finalResult.npy {5} {8} {9}\" >> bashExtensionPartClassification_{2}_parts_{3}_extensionParts_{4}_shape_{5}_pooling_{6}_classifier_{7}_numOfClass_{8}_trainingSize_{9}_classificationType.sbatch && sbatch bashExtensionPartClassification_{2}_parts_{3}_extensionParts_{4}_shape_{5}_pooling_{6}_classifier_{7}_numOfClass_{8}_trainingSize_{9}_classificationType.sbatch" ::: 10 ::: 100 ::: 10 ::: 12 ::: 8 ::: rot-mixture ::: 0 ::: 1000 ::: testing ::: 16

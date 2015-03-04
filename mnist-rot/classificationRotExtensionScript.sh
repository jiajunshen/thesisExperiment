#!/bin/bash

module load parallel

parallel "

echo \"#!/bin/bash

#SBATCH --account=pi-yaliamit
#SBATCH --ntasks=1
#SBATCH --job-name=rotExtensionPoolingClassification_{2}_parts_{3}_extensionParts_{5}_rotations_{4}_shape_{6}_spreading_{7}_pooling_{8}_classifier_{9}_numOfClass_{10}_trainingSize_{11}_classificationType
#SBATCH --output=/project/yaliamit/jiajun-master/MNIST_ROT/rotExtensionClassificationModelOutputFile/rotExtensionPoolingClassification_{2}_parts_{3}_extensionParts_{5}_rotations_{4}_shape_{6}_spreading_{7}_pooling_{8}_classifier_{9}_numOfClass_{10}_trainingSize_{11}_classificationType.out
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=32000
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rcc.jiajun.uchicago@gmail.com

module load python/2.7-2014q2
module load parallel

python trainClassification.py /project/yaliamit/jiajun-master/MNIST_ROT/rotExtensionPartsModel/rotExtensionParts_{2}_{3}_parts_{5}_rotations_{4}_shape_{6}_spreading_seed_ $ROT_MNIST/mnistROT.npy $ROT_MNIST/mnistROTLabel.npy /project/yaliamit/jiajun-master/MNIST_ROT/rotExtensionClassificationModel/rotExtensionClassification_{2}_parts_{3}_extensionParts_{5}_rotations_{4}_shape_{7}_pooling_{6}_spreading_{8}_classifier_{9}_numOfClass_{10}_trainingSize_{11}_classificationType.npy $ROT_MNIST/mnistROTTEST.npy $ROT_MNIST/mnistROTLABELTEST.npy {1} {8} {9} {10} ./finalResult.txt {7} {10} {11}\" >> bashRotExtensionPartClassification_{2}_parts_{3}_extensionParts_{5}_rotations_{4}_shape_{7}_pooling_{6}_spreading_{8}_classifier_{9}_numOfClass_{10}_trainingSize_{11}_classificationType.sbatch && sbatch bashRotExtensionPartClassification_{2}_parts_{3}_extensionParts_{5}_rotations_{4}_shape_{7}_pooling_{6}_spreading_{8}_classifier_{9}_numOfClass_{10}_trainingSize_{11}_classificationType.sbatch" ::: 10 ::: 25 ::: 5 10 ::: 12 ::: 16 ::: 0 ::: 4 8 ::: svm ::: 1 ::: 1000 ::: testing ::: 16

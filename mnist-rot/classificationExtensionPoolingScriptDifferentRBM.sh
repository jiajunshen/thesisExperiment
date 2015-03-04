#!/bin/bash

module load parallel

parallel "

echo \"#!/bin/bash

#SBATCH --account=pi-yaliamit
#SBATCH --ntasks=1
#SBATCH --job-name=extensionPoolingClassification_{2}_parts_{3}_extensionParts_{4}_shape_{5}_pooling_{6}_distance_{7}_classifier_{8}_numOfClass_{9}_trainingSize_{10}_classificationType
#SBATCH --output=/project/yaliamit/jiajun-master/extensionPoolingClassificationModelOutputFile/extensionPoolingClassification_{2}_parts_{3}_extensionParts_{4}_shape_{5}_pooling_{6}_distance_{7}_classifier_{8}_numOfClass_{9}_trainingSize_{10}_classificationType.out
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=32000
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rcc.jiajun.uchicago@gmail.com

module load python/2.7-2014q2
module load parallel

python trainClassification.py /project/yaliamit/jiajun-master/extensionPoolingModel/extensionPooling_{2}_parts_{3}_extensionParts_{4}_shape_pooling_{5}_distance_{6}_seed_ /home/jiajun/.mnist/mnistTrainingData.npy /home/jiajun/.mnist/mnistTrainingLabel.npy /project/yaliamit/jiajun-master/extensionPoolingClassificationModel/extensionPoolingClassification_{2}_parts_{3}_extensionParts_{4}_shape_{5}_pooling_{6}_distance_{7}_classifier_{8}_numOfClass_{9}_trainingSize_{10}_classificationType_seed_ /home/jiajun/.mnist/mnistTestingData.npy /home/jiajun/.mnist/mnistTestingLabel.npy {1} {7} {8} /project/yaliamit/jiajun-master/finalResult/extensionPoolingClassification_{2}_parts_{3}_extensionParts_{4}_shape_{5}_pooling_{6}_distance_{7}_classifier_{8}_numOfClass_{9}_trainingSize_{10}_classificationType.npy 0 {9} {10}\" >> bashExtensionPoolingClassification_{2}_parts_{3}_extensionParts_{4}_shape_{5}_pooling_{6}_distance_{7}_classifier_{8}_numOfClass_{9}_trainingSize_{10}_classificationType.sbatch && sbatch bashExtensionPoolingClassification_{2}_parts_{3}_extensionParts_{4}_shape_{5}_pooling_{6}_distance_{7}_classifier_{8}_numOfClass_{9}_trainingSize_{10}_classificationType.sbatch" ::: 10 ::: 100 ::: 5 10 ::: 10 12 ::: 8 ::: 5 ::: mixture ::: 1 2 5 ::: 10 ::: testing

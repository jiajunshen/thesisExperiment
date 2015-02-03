#!/bin/bash

module load parallel

parallel "

echo \"#!/bin/bash

#SBATCH --account=pi-yaliamit
#SBATCH --ntasks=1
#SBATCH --job-name=plainPartsClassification_{2}_Parts_{3}_pooling_{4}_classifier_{5}_numOfClass_{6}_trainingSize_{7}_classificationType
#SBATCH --output=/project/yaliamit/jiajun-master/plainClassificationModelOutputFile/plainPartsClassification_{2}_Parts_{3}_pooling_{4}_classifier_{5}_numOfClass_{6}_trainingSize_{7}_classificationType.out
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=32000
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rcc.jiajun.uchicago@gmail.com

module load python/2.7-2014q2
module load parallel

python trainClassification.py ./plainPartsModel/plainParts_{2}_parts_seed_ /home/jiajun/.mnist/mnistTrainingData.npy /home/jiajun/.mnist/mnistTrainingLabel.npy /project/yaliamit/jiajun-master/plainClassificationModel/plainParts_{2}_pooling_{3}_trainingSize_{6}_classifier_{4}_numOfClass_{5}_testingType_{7}_seed_ /home/jiajun/.mnist/mnistTestingData.npy /home/jiajun/.mnist/mnistTestingLabel.npy {1} {4} {5} /project/yaliamit/jiajun-master/finalResult/plainPartsClassification_{2}_pooling_{3}_trainingSize_{6}_classifier_{4}_numOfClass_{5}_testingType_{7}.npy {3} {6} {7}\" >> bashPlainPartsClassification_{2}_Parts_{3}_pooling_{4}_classifier_{5}_numOfClass_{6}_trainingSize_{7}_classificationType.sbatch && sbatch bashPlainPartsClassification_{2}_Parts_{3}_pooling_{4}_classifier_{5}_numOfClass_{6}_trainingSize_{7}_classificationType.sbatch" ::: 10 ::: 100 ::: 4 ::: svm ::: 0 ::: 100 ::: testing

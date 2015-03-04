#!/bin/bash

module load parallel

parallel "

echo \"#!/bin/bash

#SBATCH --account=pi-yaliamit
#SBATCH --ntasks=1
#SBATCH --job-name=MNIST-ROT-plainPartsClassificationWithRotObject_{2}_Parts_{3}_pooling_{4}_classifier_{5}_numOfClass_{6}_trainingSize_{7}_classificationType_{8}_objectModelRotation
#SBATCH --output=/project/yaliamit/jiajun-master/MNIST_ROT/plainPartsClassificationModelOutputFile/plainPartsClassificationWithRotObject_{2}_Parts_{3}_pooling_{4}_classifier_{5}_numOfClass_{6}_trainingSize_{7}_classificationType_{8}_objectModelRotation.out
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=32000
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rcc.jiajun.uchicago@gmail.com

module load python/2.7-2014q2
module load parallel

python trainClassification.py /project/yaliamit/jiajun-master/MNIST_ROT/plainPartsModel/plainParts_{2}_parts_seed_ $ROT_MNIST/mnistROT.npy $ROT_MNIST/mnistROTLabel.npy /project/yaliamit/jiajun-master/MNIST_ROT/plainPartsClassificationModel/plainParts_{2}_pooling_{3}_trainingSize_{6}_classifier_{4}_numOfClass_{5}_testingType_{7}_seed_ $ROT_MNIST/mnistROTTEST.npy $ROT_MNIST/mnistROTLABELTEST.npy {1} {4} {5} {8} /project/yaliamit/jiajun-master/MNIST_ROT/finalResult/plainPartsClassification_{2}_pooling_{3}_trainingSize_{6}_classifier_{4}_numOfClass_{5}_testingType_{7}.npy {3} {6} {7}\" >> bashPlainPartsClassification_{2}_Parts_{3}_pooling_{4}_classifier_{5}_numOfClass_{6}_trainingSize_{7}_classificationType.sbatch && sbatch bashPlainPartsClassification_{2}_Parts_{3}_pooling_{4}_classifier_{5}_numOfClass_{6}_trainingSize_{7}_classificationType.sbatch" ::: 10 ::: 100 ::: 4 8 ::: rot-mixture ::: 1 ::: 1000 ::: testing ::: 16

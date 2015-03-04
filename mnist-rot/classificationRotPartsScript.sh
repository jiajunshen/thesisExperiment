#!/bin/bash

module load parallel

parallel "

echo \"#!/bin/bash

#SBATCH --account=pi-yaliamit
#SBATCH --ntasks=1
#SBATCH --job-name=rotPartClassification_{2}_parts_{3}_orientation_{4}_spreading_{5}_pooling_{6}_classifier_{7}_numOfClass_{8}_trainingSize_{9}_classificationType_{10}_objectModelRotation
#SBATCH --output=/project/yaliamit/jiajun-master/MNIST_ROT/rotPartsClassificationModelOutputFile/rotPartClassification_{2}_parts_{3}_orientation_{4}_spreading_{5}_pooling_{6}_classifier_{7}_numOfClass_{8}_trainingSize_{9}_classificationType_{10}_objectModelRotation_new1.out
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=32000
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rcc.jiajun.uchicago@gmail.com

module load python/2.7-2014q2
module load parallel

python trainClassification.py /project/yaliamit/jiajun-master/MNIST_ROT/rotPartsModel/rotParts_{2}_parts_{3}_orientation_{4}_spreading_seed_ $ROT_MNIST/mnistROT.npy $ROT_MNIST/mnistROTLabel.npy /project/yaliamit/jiajun-master/MNIST_ROT/rotPartsClassificationModel/rotPartClassification_{2}_parts_{3}_orientation_{4}_spreading_{5}_pooling_{6}_classifier_{7}_numOfClass_{8}_trainingSize_{9}_classificationType_{10}_objectModelRotation_new1.npy $ROT_MNIST/mnistROTTEST.npy $ROT_MNIST/mnistROTLABELTEST.npy {1} {6} {7} {10} ./finalResult.npy {5} {8} {9}\" >> bashRotPartClassification_{2}_parts_{3}_orientation_{4}_spreading_{5}_pooling_{6}_classifier_{7}_numOfClass_{8}_trainingSize_{9}_classificationType_{10}_objectModelRotation_new1.sbatch && sbatch bashRotPartClassification_{2}_parts_{3}_orientation_{4}_spreading_{5}_pooling_{6}_classifier_{7}_numOfClass_{8}_trainingSize_{9}_classificationType_{10}_objectModelRotation_new1.sbatch" ::: 10 ::: 50 ::: 16 ::: 0 ::: 4 8 ::: mixture ::: 1 ::: 1000 ::: testing ::: 16

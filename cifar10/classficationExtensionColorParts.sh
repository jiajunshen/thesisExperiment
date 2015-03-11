#!/bin/bash

module load parallel
parallel "

echo \"#!/bin/bash

#SBATCH --account=pi-yaliamit
#SBATCH --ntasks=1
#SBATCH --job-name=trainExtensionColorPartsClassification_{1}part_5_extension_{2}classifier_{3}numofClass_{4}numofOrientation_{5}pooling
#SBATCH --output=trainExtensionColorPartsClassification_{1}part_5_extension_{2}classifier_{3}numofClass_{4}numofOrientation_{5}pooling.out
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=32000
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rcc.jiajun.uchicago@gmail.com

module load python/2.7-2014q2
module load parallel

python trainClassification.py trainExtensionColorParts_200part_5extention_12size.npy {2} {3} {4} {5}\" > bashExtensionColorPartsClassification_{1}_5_extension_classifier{2}_numofclass{3}_numofOrientation{4}_pooling{5}.sbatch && sbatch bashExtensionColorPartsClassification_{1}_5_extension_classifier{2}_numofclass{3}_numofOrientation{4}_pooling{5}.sbatch" ::: 200 ::: svm ::: 1 ::: 8 ::: 2 4 8

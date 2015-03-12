#!/bin/bash

module load parallel
parallel "

echo \"#!/bin/bash

#SBATCH --account=pi-yaliamit
#SBATCH --ntasks=1
#SBATCH --job-name=trainRotatableColorPartsClassification_{1}part_8_rotation_{2}classifier_{3}numofClass_{4}numofOrientation_{5}pooling
#SBATCH --output=trainRotatableColorPartsClassification_{1}part_8_rotation_{2}classifier_{3}numofClass_{4}numofOrientation_{5}pooling.out
#SBATCH --time=24:00:00
#SBATCH --partition=bigmem
#SBATCH --mem-per-cpu=16000
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rcc.jiajun.uchicago@gmail.com

module load python/2.7-2014q2
module load parallel

python trainClassification.py trainRotatableColorParts_{1}Parts_8Rotations.npy {2} {3} {4} {5}\" > bashRotatableColorParts_{1}_8_rotation_classifier{2}_numofclass{3}_numofOrientation{4}_pooling{5}.sbatch && sbatch bashRotatableColorParts_{1}_8_rotation_classifier{2}_numofclass{3}_numofOrientation{4}_pooling{5}.sbatch" ::: 50 ::: rot-mixture ::: 1 ::: 8 ::: 4

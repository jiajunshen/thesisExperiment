#!/bin/bash

module load parallel
parallel "

echo \"#!/bin/bash

#SBATCH --account=pi-yaliamit
#SBATCH --ntasks=1
#SBATCH --job-name=trainRegularColorPartsClassification_{1}part_{2}classifier_{3}numofClass_{4}numofOrientation_{5}pooling
#SBATCH --output=trainRegularColorPartsClassification_{1}part_{2}classifier_{3}numofClass_{4}numofOrientation_{5}pooling.out
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=32000
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rcc.jiajun.uchicago@gmail.com

module load python/2.7-2014q2
module load parallel

python trainClassification.py trainRegularColorParts_{1}.npy {2} {3} {4} {5}\" > bashRegularColorParts_{1}_classifier{2}_numofclass{3}_numofOrientation{4}_pooling{5}.sbatch && sbatch bashRegularColorParts_{1}_classifier{2}_numofclass{3}_numofOrientation{4}_pooling{5}.sbatch" ::: 200 400 ::: mixture ::: 1 5 10 ::: 0 ::: 4 8 

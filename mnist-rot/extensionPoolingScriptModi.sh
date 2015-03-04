#!/bin/bash

module load parallel

parallel "
sbatch bashExtensionPooling_{2}_{3}_parts_{4}_shape_pooling_{5}_distance_{6}_seed_{1}.sbatch" ::: {2..4} ::: 100 ::: 5 10 ::: 10 12 ::: 4 8 ::: 5 10

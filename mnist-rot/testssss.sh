#!/bin/bash

module load parallel

parallel "

echo \"#!/bin/bash
python trainExtensionPoolingModel.py\" >> bashExtensionPooling_{2}_{3}_parts_{4}_shape_pooling_{5}_distance_{6}_seed_{1}_1.sbatch && sbatch bashExtensionPooling_{2}_{3}_parts_{4}_shape_pooling_{5}_distance_{6}_seed_{1}_1.sbatch" ::: 1 ::: 100 ::: 5 ::: 10 ::: 4 ::: 5

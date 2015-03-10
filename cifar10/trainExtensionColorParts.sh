#!/bin/bash

parallel "python trainExtensionColorParts.py trainRegularColorParts_{1}.npy {1} {2} {3} trainExtensionColorParts_{1}part_{2}extention_{3}size.npy" ::: 400 ::: 5 ::: 12

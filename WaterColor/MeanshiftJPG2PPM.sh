#!/bin/bash

imagePath="$1"
echo $imagePath
/Applications/MATLAB_R2016a.app/bin/matlab -nodesktop -nosplash -nodisplay -nojvm -r "imwrite(imread('$imagePath'), 'meanshiftTemp/input.ppm'),quit()"
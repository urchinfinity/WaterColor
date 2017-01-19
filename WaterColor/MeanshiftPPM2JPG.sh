#!/bin/bash

/Applications/MATLAB_R2016a.app/bin/matlab -nodesktop -nosplash -nodisplay -nojvm -r "imwrite(imread('meanshiftTemp/meanshift.ppm'), 'meanshiftTemp/meanshift.jpg'),quit()"
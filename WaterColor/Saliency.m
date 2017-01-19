
I = imread(imagePath);

%compute energe
map = gbvs(I);
saliencyMap = map.master_map_resized;
inverseMap = (saliencyMap * -1) + 1;
imwrite(inverseMap, 'saliencyTemp/out/saliency_RCC.png');
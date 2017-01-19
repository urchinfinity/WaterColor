#include "WaterColor.h"
#include "ColorAdjustment.h"
#include "SaliencyDistance.h"
#include "Abstraction.h"
#include "WetInWet.h"
#include "HandTremorEffect.h"
#include "EdgeDarkening.h"
#include "Granulation.h"
#include "TurblenceFlow.h"
#include "PigmentVariation.h"
#include "ToolBox.h"
#include "Debug.h"

using namespace cv;

RNG rng2;

void texture(Mat &input, Mat &output) {
	Mat texture = imread("texture/default.jpg");
	Mat textureScaled = input.clone();

    for (int y = 0; y < textureScaled.rows; y++) {
		for (int x = 0; x < textureScaled.cols; x++){
			textureScaled.at<Vec3b>(y, x) = texture.at<Vec3b>(y%texture.rows, x%texture.cols);
		}
    }
    
    output = input.clone();
    double alpha = 0.8;
    addWeighted(input, alpha, textureScaled, 1-alpha, 0.0, output);
}

void applyMedianFilter(int filterSize, Mat &input, Mat &output, int threshold) {
    int padding = filterSize/2;
    
    output = input.clone();
    Mat inputGray, inputPadded, inputPaddedGray;
    copyMakeBorder(input, inputPadded, padding, padding, padding, padding, BORDER_REPLICATE, Scalar(255,0,0));
    cvtColor(inputPadded, inputPaddedGray, CV_RGB2GRAY);
    
    uchar *intensity = new uchar[filterSize*filterSize];
    for (int y = padding; y < input.rows + padding; y++) {
        for (int x = padding; x < input.cols + padding; x++) {
            int flag = 0;
            for (int r = y - filterSize/2; r <= y + filterSize/2; r++) {
                for (int c = x - filterSize/2; c <= x + filterSize/2; c++) {
                    intensity[flag++] = inputPaddedGray.at<uchar>(r, c);
                }
            }
            sort(intensity, intensity+filterSize*filterSize);
            uchar median = intensity[filterSize*filterSize/2];
            
            if (abs(inputPaddedGray.at<uchar>(y, x)-median) < threshold)
                continue;
            
            for (int r = y - filterSize/2; r <= y + filterSize/2; r++) {
                for (int c = x - filterSize/2; c <= x + filterSize/2; c++) {
                    if (inputPaddedGray.at<uchar>(r, c) == median) {
                        output.at<Vec3b>(y-padding, x-padding) = inputPadded.at<Vec3b>(r, c);
                    }
                }
            }
        }
    }
    delete []intensity;
}

void applyErosion(Mat &input, Mat &output) {
    output = input.clone();
    int padding = 1, filterSize = 3;
    Mat inputPadded;
    copyMakeBorder(input, inputPadded, padding, padding, padding, padding, BORDER_REPLICATE, Scalar(255,0,0));
    
    int kernel[3][3] = {{0, 1, 0}, {1, 1, 1}, {0, 1, 0}};
    
    for (int y = padding; y < input.rows + padding; y++) {
        for (int x = padding; x < input.cols + padding; x++) {
            bool isEqual = true;
            for (int r = -filterSize/2; r <= filterSize/2; r++) {
                for (int c = -filterSize/2; c <= filterSize/2; c++) {
                    if (kernel[r+filterSize/2][c+filterSize/2] == 1 && inputPadded.at<uchar>(y+r, x+c) != 255) {
                        isEqual = false;
                    }
                }
            }
            
            if (isEqual)
                output.at<uchar>(y-padding, x-padding) = 255;
            else {
                output.at<uchar>(y-padding, x-padding) = 0;
            }
        }
    }
}

void applyMeanFilter(Mat &input, Mat &output, Mat &meanShift) {
    output = input.clone();
    
    int padding = 1, filterSize = 3;
    
    Mat inputPadded, meanShiftPadded;
    copyMakeBorder(input, inputPadded, padding, padding, padding, padding, BORDER_REPLICATE, Scalar(255,0,0));
    copyMakeBorder(meanShift, meanShiftPadded, padding, padding, padding, padding, BORDER_REPLICATE, Scalar(255,0,0));
    
    for (int y = padding; y < input.rows + padding; y++) {
        for (int x = padding; x < input.cols + padding; x++) {

            int blue = 0;
            int green = 0;
            int red = 0;
            int count = 0;
            
            for (int r = y - filterSize/2; r <= y + filterSize/2; r++) {
                for (int c = x - filterSize/2; c <= x + filterSize/2; c++) {
                    if (meanShiftPadded.at<Vec3b>(r, c) == meanShiftPadded.at<Vec3b>(y, x)) {
                        Vec3b color = inputPadded.at<Vec3b>(r, c);
                        blue += int(color.val[0]);
                        green += int(color.val[1]);
                        red += int(color.val[2]);
                        count++;
                    }
                }
            }
            if (count == 0) count++;
            
            output.at<Vec3b>(y-padding, x-padding) = Vec3b(uchar(blue/count), uchar(green/count), uchar(red/count));
        }
    }
}

void WaterColor::deal(Mat &input, Mat &output,string inputName, string SPName) {
    
	output = input.clone();

	//Color transform
	Debug() << "color transform...";
    
	Mat myColorTransform;
	ColorAdjustment colorAdjustment;
//    char paintingStyle[] = "all";
//	colorAdjustment.chooseOneStyle(paintingStyle);
	colorAdjustment.deal(input, myColorTransform);
	imwrite("process/colorTransform.jpg", myColorTransform);
	
    
	//Simplify transform
//	Debug() << "saliency...";
//    
//	Mat mySaliency, myDis;
//	SaliencyDistance saliencyDiatance;
//	saliencyDiatance.deal(myColorTransform, mySaliency, myDis);
//    
//	imwrite("process/dis.jpg", myDis*255);
//	imwrite("process/saliency.jpg", mySaliency);

    
	//Abstraciton
	Debug() << "abstraction...";
    
	Mat myAbstraction;
	Abstraction abstraction;
	abstraction.deal(inputName, SPName, myColorTransform, myAbstraction);
	imwrite("process/abstraciton.jpg", myAbstraction);


	//Add wet-in-wet effect
	Debug() << "wet in wet...";
    
    WetInWet wetInWet;
    Mat myWetInWet, myCanny;
	wetInWet.deal(SPName, input, myAbstraction, myWetInWet, myCanny);
	imwrite("process/wetinwet.jpg", myWetInWet);

    
    //remove noise
    Mat myDenoising;
    fastNlMeansDenoisingColored(myWetInWet, myDenoising);
    imwrite("process/denoising.jpg", myDenoising);
    
    Mat myMedian;
    applyMedianFilter(3, myDenoising, myMedian, 10);
    imwrite("process/medianFilterDenoising.jpg", myMedian);
    
    
    //test meanshift for current output
    char command[100];
    sprintf(command, "./MeanshiftJPG2PPM.sh process/medianFilterDenoising.jpg");
    system(command);
    
    ofstream myfile;
    myfile.open ("edison/edsScript2.eds");
    
    myfile << "SpatialBandwidth = 7;" << endl;
    myfile << "RangeBandwidth = 6.5;" << endl;
    myfile << "MinimumRegionArea = 20;" << endl;
    myfile << "Speedup = MEDIUM;" << endl;
    myfile << "Load('meanshiftTemp/input.ppm', IMAGE);" << endl;
    myfile << "Segment;" << endl;
    myfile << "Save('meanshiftTemp/meanshift.ppm', PPM, SEGM_IMAGE);" << endl;
    
    myfile.close();
    
    system("edison/edison edison/edsScript2.eds");
    system("./MeanshiftPPM2JPG.sh");
    
    
    //segment
    Mat meanshift2 = imread("meanshiftTemp/meanshift.jpg");
    Mat mySegment;
    applyMedianFilter(9, meanshift2, mySegment, 0);
    imwrite("process/segmentImage.png", mySegment);
    
    //get layers
    int layers = 0;
    int intensity[256];
    Mat mySegmentGray;
    cvtColor(mySegment, mySegmentGray, CV_RGB2GRAY);
    
    for (int i = 0; i < 256; i++)
        intensity[i] = 0;

    for (int y = 0; y < mySegmentGray.rows; y++) {
        for (int x = 0; x < mySegmentGray.cols; x++) {
            if (intensity[int(mySegmentGray.at<uchar>(y, x))] == 0) {
                layers++;
                intensity[int(mySegmentGray.at<uchar>(y, x))] = 1;
            }
        }
    }
    
    Mat myVariation = myMedian.clone();
    for (int i = 0; i < 256; i++) {
        if (intensity[i] == 1) {
            int count = 0;
            for (int y = 0; y < mySegmentGray.rows; y++)
                for (int x = 0; x < mySegmentGray.cols; x++)
                    if (int(mySegmentGray.at<uchar>(y, x)) == i)
                        count++;
            
            if (count < mySegmentGray.rows * 3)
                continue;
                
            Mat layerImage = mySegmentGray.clone();
            
            for (int y = 0; y < mySegmentGray.rows; y++)
                for (int x = 0; x < mySegmentGray.cols; x++)
                    layerImage.at<uchar>(y, x) = abs(int(mySegmentGray.at<uchar>(y, x))-i) < 10 ? 255 : 0;
            
            Mat layerImageErosionIn = layerImage.clone();
            Mat layerImageErosionOut;
            
            Mat labelImage = layerImage.clone();
            for (int y = 0; y < layerImageErosionIn.rows; y++) {
                for (int x = 0; x < layerImageErosionIn.cols; x++) {
                    labelImage.at<uchar>(y, x) = 0;
                }
            }
            
            for (int k = 0; k < 30; k++) {
                applyErosion(layerImageErosionIn, layerImageErosionOut);
                layerImageErosionIn = layerImageErosionOut.clone();
                
                for (int y = 0; y < layerImageErosionOut.rows; y++) {
                    for (int x = 0; x < layerImageErosionOut.cols; x++) {
                        if (layerImageErosionOut.at<uchar>(y, x) == 255) {
//                            labelImage.at<uchar>(y, x) = (k/10+1) * 5;
//                            labelImage.at<uchar>(y, x) = abs(int(rng2.gaussian((k/10+1) * 5)));
                            labelImage.at<uchar>(y, x) = k+1;
                        }
                    }
                }
            }
            
            char layerName[30];
            sprintf(layerName, "process/layer/%d.png", i);
            imwrite(layerName, layerImage);
            
            
            char layerErosionName[35];
            sprintf(layerErosionName, "process/layer/e_40%d.png", i);
            imwrite(layerErosionName, layerImageErosionOut);
            
            cout << layerErosionName << endl;
            
            
            for (int y = 0; y < layerImageErosionOut.rows; y++) {
                for (int x = 0; x < layerImageErosionOut.cols; x++) {
                    if (labelImage.at<uchar>(y, x) == 0)
                        continue;
                    
                    int pigment = -1;
                    int pigmentRnd = min(abs(int(rng2.gaussian(15))), 20);
                    
                    if (labelImage.at<uchar>(y, x) > 20) {
                        if (pigmentRnd > 16)
                            pigment = pigmentRnd-15;
                    } else if (labelImage.at<uchar>(y, x) > 10) {
                        if (pigmentRnd > 14)
                            pigment = pigmentRnd-8;
                    } else {
                        if (pigmentRnd > 9)
                            pigment = pigmentRnd-3;
                    }
                    
                    if (pigment == -1) {
                        pigment = labelImage.at<uchar>(y, x)/3;
                        
                        myVariation.at<Vec3b>(y, x).val[0] = min(max(int(myVariation.at<Vec3b>(y, x).val[0]),
                                                                     myMedian.at<Vec3b>(y, x).val[0]+pigment), 255);
                        myVariation.at<Vec3b>(y, x).val[1] = min(max(int(myVariation.at<Vec3b>(y, x).val[1]),
                                                                     myMedian.at<Vec3b>(y, x).val[1]+pigment), 255);
                        myVariation.at<Vec3b>(y, x).val[2] = min(max(int(myVariation.at<Vec3b>(y, x).val[2]),
                                                                     myMedian.at<Vec3b>(y, x).val[2]+pigment), 255);
                    }
                    
                    else {
                        myVariation.at<Vec3b>(y, x).val[0] = max(min(int(myVariation.at<Vec3b>(y, x).val[0]),
                                                                     myMedian.at<Vec3b>(y, x).val[0]-pigment), 0);
                        myVariation.at<Vec3b>(y, x).val[1] = max(min(int(myVariation.at<Vec3b>(y, x).val[1]),
                                                                     myMedian.at<Vec3b>(y, x).val[1]-pigment), 0);
                        myVariation.at<Vec3b>(y, x).val[2] = max(min(int(myVariation.at<Vec3b>(y, x).val[2]),
                                                                     myMedian.at<Vec3b>(y, x).val[2]-pigment), 0);
                    }
                }
            }
            
            i += 10;
        }
    }
    Mat myVariationMean;
    applyMeanFilter(myVariation, myVariationMean, mySegment);
    imwrite("process/variation.jpg", myVariationMean);
    
    
//	Mat myHandTremor;
//	HandTremorEffect handTremoeEffect;
//	handTremoeEffect.deal(myWetInWet, myHandTremor, myCanny);
//	imwrite("process/handtremor.jpg", myHandTremor);


    //apply edge darkening
	EdgeDarkening edgeDarkening;
    Mat myEdgeDarkening;
    edgeDarkening.deal(myWetInWet, myEdgeDarkening, SPName);
    
    imwrite("process/edgeDarkening.jpg", myEdgeDarkening);
    cout << "segment finished" << endl;
    
//
//	Granulation granulation;
//	granulation.deal(output, output);
//	
//	TurblenceFlow turblenceFlow;
//	turblenceFlow.deal(output, output);
//	
//	//After deal
//	PigmentVariation pigmentVariation;
//	pigmentVariation.deal(output, output);
//
//	//DoubleToImage(output, output);
//	//DoubleToImage(output, output);
//	output.convertTo(output, CV_8UC3, 255.0);

	
	//Mat myWetInWet = imread("process/wetinwet.jpg");
	Mat myTexture;
	texture(myVariationMean, myTexture);
	Debug() << myTexture.channels();
	imwrite("process/texture.png", myTexture);
    
	output = myTexture;
}

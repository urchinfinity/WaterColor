#include "Abstraction.h"
#include "ToolBox.h"
#include "Debug.h"
#include "include/meanshift/MeanShift.h"

#include <iostream>
#include <fstream>
using namespace std;
using namespace cv;


//ver 3.0
void meanShift(string inputName, Mat &dst) {
//    imwrite("meanshiftTemp/input.ppm", imread(inputName));
    char command[100];
    sprintf(command, "./MeanshiftJPG2PPM.sh input/%s", inputName.c_str());
    system(command);
    
    ofstream myfile;
    myfile.open ("edison/edsScript.eds");
    
    myfile << "SpatialBandwidth = 7;" << endl;
    myfile << "RangeBandwidth = 6.5;" << endl;
    myfile << "MinimumRegionArea = 20;" << endl;
    myfile << "Speedup = MEDIUM;" << endl;
    myfile << "Load('meanshiftTemp/input.ppm', IMAGE);" << endl;
    myfile << "Segment;" << endl;
    myfile << "Save('meanshiftTemp/meanshift.ppm', PPM, SEGM_IMAGE);" << endl;
    
    myfile.close();
    
    system("edison/edison edison/edsScript.eds");
    system("./MeanshiftPPM2JPG.sh");
    
    dst = imread("meanshiftTemp/meanshift.jpg");
    Debug() << "meanshift...";
}



Vec3b getInMean(Mat &src,Mat &meanshift, int x, int y){
	int count = 0;
	int size = 2;///
	Scalar sum = Scalar::all(0);
	for (int i = x - size; i <= x + size; i++){
		for (int j = y - size; j <= y + size; j++){
			if (dataAt<uchar>(meanshift, x, y) == dataAt<uchar>(meanshift, i, j)){
				sum += Scalar(dataAt<Vec3b>(src, i, j));
				count++;
			}
		}
	}
	Scalar ans = divVec(sum, float(count));
	//Debug() <<"ans:"<< ans<<" "<<"sum: "<<sum<<count;
	//Debug().pause();
	return Vec3b(ans[0],ans[1],ans[2]);
}
int clamp(int a, int x, int y){
	if (a < x) return x;
	if (a > y) return y;
	return a;
}
bool haveColorDifference(Mat &dis,Mat &meanshift,int x,int y){
	float d = dataAt<float>(dis, x, y);
	int size = clamp(5 * 2 * (d + 0.3), 4, 9) ;///
	for (int i = x - size; i <= x + size; i++){
		for (int j = y - size; j <= y + size; j++) if (ArraySpace::inMap(dis,myPoint(i,j))){
			if (dataAt<uchar>(meanshift, x, y) != dataAt<uchar>(meanshift, i, j) &&
				fabs(dataAt<float>(dis, i, j) - d) < 0.3*d){
				
				return true;
			}
		}
	}
	return false;
}
Vec3b getOutMean(Mat &src, Mat& myMeanShift,int x, int y, int size){
	Scalar sum = Scalar::all(0);
	int count = 0;
	for (int i = x - size; i <= x + size; i++){
		for (int j = y - size; j <= y + size; j++) if (ArraySpace::inMap(src, myPoint(i, j))){
			if (dataAt<uchar>(myMeanShift, x, y) == dataAt<uchar>(myMeanShift, i, j)){
				sum += Scalar(dataAt<Vec3b>(src, i, j));
				count++;
			}
		}
	}
	Scalar ans = divVec(sum, float(count));
	//Debug() << ans;
	//Debug().pause();
	return Vec3b(ans[0], ans[1], ans[2]);
}

void Abstraction::deal(string inputName, string SPName, Mat &src, Mat &dst) {

	Debug().setStatus(StdOut);
    
    Mat mySceneParse = imread("sceneParse/"+SPName);
    imwrite("process/sceneparse.png", mySceneParse);
    
    Mat myMeanShift;
    meanShift(inputName, myMeanShift);
    imwrite("process/meanshift.png", myMeanShift);
    
    int cat = 0;
    int intensity[256];
    
    for (int i = 0; i < 256; i++)
        intensity[i] = 0;
    
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            if (intensity[int(mySceneParse.at<Vec3b>(y, x).val[0])] == 0) {
                cat++;
                intensity[int(mySceneParse.at<Vec3b>(y, x).val[0])] = 1;
            }
        }
    }
    
    Debug() << "	scene parsing...";
    Debug() << "	category: " << cat;
    
    int *depthID = new int[cat];
    for (int i = 255, count = 0; i >= 0 && count < cat; i--)
        if (intensity[i] == 1)
            depthID[count++] = i;
    
	Debug() << "	apply mean filter...";
    
    dst = src.clone();
    Mat paddedSceneParse, paddedMeanShift, paddedDst;
    int padding = 30;
    copyMakeBorder(mySceneParse, paddedSceneParse, padding, padding, padding, padding, BORDER_REPLICATE, Scalar(255,0,0));
    copyMakeBorder(myMeanShift, paddedMeanShift, padding, padding, padding, padding, BORDER_REPLICATE, Scalar(255,0,0));
    copyMakeBorder(src, paddedDst, padding, padding, padding, padding, BORDER_REPLICATE, Scalar(255,0,0));
    
    Mat paddedGraySrc;
    cvtColor(paddedDst, paddedGraySrc, CV_RGB2GRAY);
    
    Mat myDepth = mySceneParse.clone();
    
    for (int y = padding; y < src.rows + padding; y++) {
        for (int x = padding; x < src.cols + padding; x++) {
            //find pixel depth
            int depth = -1;
            for (int i = 0; i < cat; i++)
                if (int(paddedSceneParse.at<Vec3b>(y, x).val[0]) == depthID[i])
                    depth = i;
            
            myDepth.at<uchar>(y-padding, x-padding) = depth;
            
            int filterSize = 0;
            if (depth <= 2)
                filterSize = 21;
            else if (depth <= 5)
                filterSize = 41;
            else
                filterSize = 2 * padding + 1;
            
            int blue = 0;
            int green = 0;
            int red = 0;
            int count = 0;
            for (int r = y - filterSize/2; r <= y + filterSize/2; r++) {
                for (int c = x - filterSize/2; c <= x + filterSize/2; c++) {
                    if (paddedMeanShift.at<Vec3b>(r, c) == paddedMeanShift.at<Vec3b>(y, x)) {
                        Vec3b color = paddedDst.at<Vec3b>(r, c);
                        blue += int(color.val[0]);
                        green += int(color.val[1]);
                        red += int(color.val[2]);
                        count++;
                    }
                }
            }
            if (count == 0) count++;
            
            dst.at<Vec3b>(y-padding, x-padding) = Vec3b(uchar(blue/count), uchar(green/count), uchar(red/count));
        }
    }
    
    imwrite("process/depth.png", myDepth);
  
    Debug() << "	complete";
}

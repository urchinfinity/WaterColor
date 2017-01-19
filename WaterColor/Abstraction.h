#ifndef ABSTRECTION_H
#define ABSTRECTION_H

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class Abstraction {
public:
	void deal(const Mat &input, Mat &output);
	// void deal(const Mat& src, Mat& mySaliency, Mat& myDis, Mat& dst);
	void deal(string inputName, string SPName, Mat& src, Mat& dst);
};

#endif

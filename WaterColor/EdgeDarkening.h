#ifndef EDGEDARKENING_H
#define EDGEDARKENING_H

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class EdgeDarkening {
public:
	void deal(const cv::Mat &input, cv::Mat &output, string SPName);
};


#endif

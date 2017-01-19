#ifndef WETINWET_H
#define WETINWET_H

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class WetInWet
{
public:
	void deal(cv::Mat& myAbstraction, cv::Mat& dst);
    void deal(string SPName, cv::Mat &src, cv::Mat& myAbstraction, cv::Mat& dst, cv::Mat& myCanny);
};


#endif

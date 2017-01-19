#ifndef WATERCOLOR_H
#define WATERCOLOR_H

#include <opencv2/opencv.hpp>

using namespace std;

class WaterColor {
    public:
        void deal(cv::Mat &input, cv::Mat &output, string inputName, string SPName);
};


#endif

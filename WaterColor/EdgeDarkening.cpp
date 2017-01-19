#include "EdgeDarkening.h"

using namespace std;
using namespace cv;

void EdgeDarkening::deal(const Mat &input, Mat &output, string SPName) {
	output = input.clone();

    Mat mySceneParse = imread("sceneParse/"+SPName);
    Mat spCanny;
    Canny(mySceneParse, spCanny, 60, 120);
    imwrite("process/sceneParseCanny.png", spCanny);
    
    
    for (int y = 0; y < input.rows; y++) {
        for (int x = 0; x < input.cols; x++) {
            if (spCanny.at<uchar>(y, x) == 255) {
                output.at<Vec3b>(y, x).val[0] = max(output.at<Vec3b>(y, x).val[0]-200, 0);
                output.at<Vec3b>(y, x).val[1] = max(output.at<Vec3b>(y, x).val[1]-200, 0);
                output.at<Vec3b>(y, x).val[2] = max(output.at<Vec3b>(y, x).val[2]-200, 0);
            }
        }
    }
}

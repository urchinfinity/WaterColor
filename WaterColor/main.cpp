#include "WaterColor.h"
#include "ToolBox.h"
#include "Debug.h"

#include <opencv2/opencv.hpp>

using namespace cv;


int main(int argc,char *argv[]){  
    //file dir
    string inName = "ADE_train_00000001.jpg";
    string SPName = "ADE_train_00000001.png";
    string inDir = "input/" + inName;
    string outDir = "output/" + inName;
    
    if (argc == 3) {
        inDir = string(argv[1]);
        outDir = string(argv[2]);
    }
    
    //read picture
    Mat input = imread(inDir),output;
    Debug() << "img size: " << input.rows << " " << input.cols;
    
    //process
    WaterColor watercolor;
    imwrite("process/input.jpg", input);
    watercolor.deal(input, output, inName, SPName);
    
    //show & store result
    imshow("output", output);
    imwrite(outDir, output);
    imwrite("process/output.png", output);
    waitKey();
    
    return 0;
}

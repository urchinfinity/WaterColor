#include <algorithm>

#include "WetInWet.h"
#include "ToolBox.h"

using namespace cv;
using namespace std;

RNG rng;
int hKernel = 7;
double featherH[15][15];
double featherV[15][15];

void getCanny(Mat &src, Mat &dst) {
	Mat img = src;
	Mat imgCanny;
	Mat imgGray;
	cvtColor(img, imgGray, CV_RGB2GRAY);
//	Mat imgBlur;
//	blur(imgGray, imgBlur, Size(5, 5));
	Canny(img, imgCanny, 60, 120);
	dst = imgCanny;
	imwrite("process/canny.jpg", imgCanny);
}

#define CENTER 0
#define INNER_BORDER 1
#define OUTER_BORDER 2

double getFeatherWeight(int length, int pos, int type) {
    double      center[8] = {0.15, 0.3, 0.45, 0.60, 0.70, 0.80, 0.90, 1.0};
    double innerBorder[8] = {0.05, 0.2, 0.35, 0.45, 0.55, 0.65, 0.75, 0.8};
    double outerBorder[8] = {0.00, 0.1, 0.20, 0.25, 0.30, 0.40, 0.50, 0.6};
    
    int dropID[7] = {4, 6, 2, 0, 3, 5, 1};
    
    for (int dropCount = 0; dropCount < 7-length; dropCount++) {
        center[dropID[dropCount]] = -1;
        innerBorder[dropID[dropCount]] = -1;
        outerBorder[dropID[dropCount]] = -1;
    }
    
    switch (type) {
        case CENTER:
            for (int i = 0, count = -1; i < 8; i++) {
                if (center[i] != -1)
                    count++;
                
                if (count == pos)
                    return center[i];
            }
        case INNER_BORDER:
            for (int i = 0, count = -1; i < 8; i++) {
                if (innerBorder[i] != -1)
                    count++;
                
                if (count == pos)
                    return innerBorder[i];
            }
        case OUTER_BORDER:
            for (int i = 0, count = -1; i < 8; i++) {
                if (outerBorder[i] != -1)
                    count++;
                
                if (count == pos)
                    return outerBorder[i];
            }
        default:
            return -1;
    }
}

void WetInWet::deal(string SPName, Mat &src, Mat &myAbstraction, Mat &dst, Mat &myCanny) {
    
    Mat myMeanShift = imread("sceneParse/"+SPName);
    
    dst = myAbstraction.clone();
    getCanny(src, myCanny);
    
    Mat paddedCanny, paddedDst, paddedSrc, paddedGraySrc;
    copyMakeBorder(myCanny, paddedCanny, hKernel, hKernel, hKernel, hKernel, BORDER_REPLICATE, Scalar(255,0,0));
    
    copyMakeBorder(myAbstraction, paddedSrc, hKernel, hKernel, hKernel, hKernel, BORDER_REPLICATE, Scalar(255,0,0));
    cvtColor(paddedSrc, paddedGraySrc, CV_RGB2GRAY);
    
    Mat myDepth = imread("process/depth.png");
    
    for (int y = 0; y < dst.rows; y++) {
        for (int x = 0; x < dst.cols; x++) {
            if (myCanny.at<uchar>(y, x) == 0)
                dst.at<Vec3b>(y, x) = Vec3b(0, 0, 0);
            else
                dst.at<Vec3b>(y, x) = myAbstraction.at<Vec3b>(y, x);
        }
    }
    
    copyMakeBorder(dst, paddedDst, hKernel, hKernel, hKernel, hKernel, BORDER_REPLICATE, Scalar(255,0,0));
    
    double weights[paddedDst.rows][paddedDst.cols];
    Mat colors = paddedDst.clone();
    
    for (int y = 0; y < paddedDst.rows; y++) {
        for (int x = 0; x < paddedDst.cols; x++) {
            weights[y][x] = 0;
            colors.at<Vec3b>(y, x) = Vec3b(0, 0, 0);
        }
    }
    
    
    for (int y = hKernel; y < dst.rows + hKernel; y++) {
        for (int x = hKernel; x < dst.cols + hKernel; x++) {
            if (dst.at<Vec3b>(y-hKernel, x-hKernel).val[0] != 0) {
                bool isTarget = true;
                for (int row = -2; row <= 2 && isTarget; row++) {
                    for (int col = -2; col <= 2 && isTarget; col++) {
                        if (y-hKernel+row < 0 || y-hKernel+row >= dst.rows + hKernel || x-hKernel+col < 0 || x-hKernel+col >= dst.cols + hKernel)
                            continue;
                        
                        if (myDepth.at<uchar>(y-hKernel+row, x-hKernel+col) <= 10)
                            isTarget = false;
                    }
                }
                
                if (!isTarget)
                    continue;
            
                int r = -1, c = -1;
                bool findNeighbor = false;
                for (int row = -1; row <= 1 && !findNeighbor; row++) {
                    for (int col = -1; col <= 1 && !findNeighbor; col++) {
                        if (row == 0 && col == 0)
                            continue;
                        
                        if (paddedCanny.at<uchar>(y+row, x+col) == 255) {
                            findNeighbor = true;
                            r = row;
                            c = col;
                        }
                    }
                }
                
                if (!findNeighbor)
                    continue;
                
                int dis = max(min(abs(int(rng.gaussian(hKernel))), hKernel-2), hKernel-5);
                int seedY, seedX;
                Vec3b curColor;
                if (paddedGraySrc.at<uchar>(y-c*dis, x+r*dis) <= paddedGraySrc.at<uchar>(y+c*dis, x-r*dis)) {
                    seedY = y+c*dis;
                    seedX = x-r*dis;
                    curColor = paddedSrc.at<Vec3b>(y-c*dis/2, x+r*dis/2);
                } else {
                    seedY = y-c*dis;
                    seedX = x+r*dis;
                    curColor = paddedSrc.at<Vec3b>(y+c*dis/2, x-r*dis/2);
                }

                
                for (int row = 0; row < 15; row++) {
                    for (int col = 0; col < 15; col++) {
                        if (c == 0 || (r != 0 && c != 0)) { //horizontal
                            for (int i = 0; i <= 0; i++) {
                                if (seedX < x) {
                                    if (row >= 5 && row <= 9 && col <= x-seedX) {
                                        double curWeight = getFeatherWeight(x-seedX, col, abs(row-7));
                                        
                                        if (weights[y+row-7+i][seedX+col] == 0) {
                                            weights[y+row-7+i][seedX+col] = curWeight;
                                            colors.at<Vec3b>(y+row-7+i, seedX+col) = curColor;
                                        } else if (weights[y+row-7+i][seedX+col] < curWeight) {
                                            paddedDst.at<Vec3b>(y+row-7+i, seedX+col) =
                                                Vec3b(255*curWeight, 255*curWeight, 255*curWeight);
                                            
                                            weights[y+row-7+i][seedX+col] = (weights[y+row-7+i][seedX+col]+curWeight)/2;
                                            colors.at<Vec3b>(y+row-7+i, seedX+col) = curColor;
                                        }
                                    }
                                } else {
                                    if (row >= 5 && row <= 9 && 14-col <= seedX-x) {
                                        double curWeight = getFeatherWeight(seedX-x, 14-col, abs(row-7));
                                        
                                        if (weights[y+row-7+i][seedX+col-14] == 0) {
                                            weights[y+row-7+i][seedX+col-14] = curWeight;
                                            colors.at<Vec3b>(y+row-7+i, seedX+col-14) = curColor;
                                        } else if (weights[y+row-7+i][seedX+col-14] < curWeight) {
                                            paddedDst.at<Vec3b>(y+row-7+i, seedX+col-14) =
                                                Vec3b(255*curWeight, 255*curWeight, 255*curWeight);
                                            
                                            weights[y+row-7+i][seedX+col-14] = (weights[y+row-7+i][seedX+col-14]+curWeight)/2;
                                            colors.at<Vec3b>(y+row-7+i, seedX+col-14) = curColor;
                                        }
                                    }
                                }
                            }
                        } else if (r == 0 || (r != 0 && c != 0)) { //vertical
                            for (int i = 0; i <= 0; i++) {
	                            if (seedY < y) {
                                    if (col >= 5 && col <= 9 && row <= y-seedY) {
                                        double curWeight = getFeatherWeight(y-seedY, row, abs(col-7));
                                        
                                        if (weights[seedY+row][x+col-7+i] == 0) {
                                            weights[seedY+row][x+col-7+i] = curWeight;
                                            colors.at<Vec3b>(seedY+row, x+col-7+i) = curColor;
                                        } else if (weights[seedY+row][x+col-7+i] < curWeight) {
	                                        paddedDst.at<Vec3b>(seedY+row, x+col-7+i) =
	                                        Vec3b(255*curWeight, 255*curWeight, 255*curWeight);
	                                        weights[seedY+row][x+col-7+i] = curWeight;
                                            colors.at<Vec3b>(seedY+row, x+col-7+i) = curColor;
	                                    }
	                                }
	                            } else {
	                                if (col >= 5 && col <= 9 && 14-row <= seedY-y) {
                                        double curWeight = getFeatherWeight(seedY-y, 14-row, abs(col-7));

                                        if (weights[seedY+row-14][x+col-7+i] == 0) {
                                            weights[seedY+row-14][x+col-7+i] = curWeight;
                                            colors.at<Vec3b>(seedY+row-14, x+col-7+i) = curColor;
                                        } else if (weights[seedY+row-14][x+col-7+i] < curWeight) {
	                                        paddedDst.at<Vec3b>(seedY+row-14, x+col-7+i) =
	                                        Vec3b(255*curWeight, 255*curWeight, 255*curWeight);
	                                        weights[seedY+row-14][x+col-7+i] = curWeight;
                                            colors.at<Vec3b>(seedY+row-14, x+col-7+i) = curColor;
	                                    }
	                                }
	                            }
	                        }
                        }
                    }
                }
            }
        }
    }
    
    imwrite("process/paddedDst.jpg", paddedDst);
            
    for (int y = hKernel; y < dst.rows + hKernel; y++) {
        for (int x = hKernel; x < dst.cols + hKernel; x++) {
            dst.at<Vec3b>(y-hKernel, x-hKernel).val[0] =
                colors.at<Vec3b>(y, x).val[0] * weights[y][x] +
                myAbstraction.at<Vec3b>(y-hKernel, x-hKernel).val[0] * (1-weights[y][x]);
            
            dst.at<Vec3b>(y-hKernel, x-hKernel).val[1] =
                colors.at<Vec3b>(y, x).val[1] * weights[y][x] +
                myAbstraction.at<Vec3b>(y-hKernel, x-hKernel).val[1] * (1-weights[y][x]);
            
            dst.at<Vec3b>(y-hKernel, x-hKernel).val[2] =
                colors.at<Vec3b>(y, x).val[2] * weights[y][x] +
                myAbstraction.at<Vec3b>(y-hKernel, x-hKernel).val[2] * (1-weights[y][x]);
        }
    }
    
    /*
	int n = src.rows, m = src.cols;
	Mat gray,mask = Mat::zeros(src.size(), CV_8U);
	dst = myAbstraction.clone();
	Debug() << "	canny...";
	getCanny(myAbstraction, myCanny);

	cvtColor(src, gray, CV_RGB2GRAY);
	myPoint** nearest = ArraySpace::newArray<myPoint>(n, m);
	myPoint* Q = new myPoint[n*m + 5];
	int **visit = ArraySpace::newArray<int>(n, m);
	
	{//getRandom
		for (int i = 0; i < n; i++){
			for (int j = 0; j < m; j++){
				if (dataAt<uchar>(myCanny, i, j) != 0){
					int kernel = 5;
					myPoint p2;
					if (randomPoint(p2,i, j, src, kernel)){
						myPoint p(i, j), p1 = getDarkestPoint(i, j, gray, kernel);
						if (dataAt<uchar>(myCanny, p1.x, p1.y) == 0 && dataAt<uchar>(gray, p2.x, p2.y) - dataAt<uchar>(gray, p1.x, p1.y)>10){
							//Debug() << "random: " << (int)dataAt<uchar>(gray, p2.x, p2.y) << " drakest: " << (int)dataAt<uchar>(gray, p1.x, p1.y);
							if (p1.dis(p) < p1.dis(p2)){
								dataAt<Vec3b>(dst, p2.x, p2.y) = dataAt<Vec3b>(dst, p1.x, p1.y);
								dataAt<uchar>(mask, p2.x, p2.y) = 1;
							}
						}
					}
				}
			}
		}
	}
	//imwrite("process/wetinwet-maskpoint.jpg", mask*255);
	{//getNearest
		int p2 = 0;
		for (int i = 0; i < n; i++)
			for (int j = 0; j < m; j++) if (dataAt<uchar>(mask, i, j) != 0){
				nearest[i][j] = myPoint(i, j);
				Q[++p2] = myPoint(i, j);
				visit[i][j] = 1;
			}
		for (int p1 = 1; p1 <= p2; p1++){
			myPoint v = Q[p1];
			for (int k = 0; k < 4; k++){
				int nx = v.x + ArraySpace::xo[k];
				int ny = v.y + ArraySpace::yo[k];
				if (ArraySpace::inMap(src, myPoint(nx, ny)) && !visit[nx][ny]){
					visit[nx][ny] = 1;
					Q[++p2] = myPoint(nx, ny);
					nearest[nx][ny] = nearest[v.x][v.y];
				}
			}
		}

	}

	{//Debug
		Mat test=Mat::zeros(src.size(),CV_8UC3);
		for (int i = 0; i < n; i++)
			for (int j = 0; j < m; j++){
				if (dataAt<uchar>(mask, i, j) != 0){
					dataAt<Vec3b>(test, i, j) = Vec3b(255, 255, 255);
				}
				if (dataAt<uchar>(myCanny, i, j) != 0){
					dataAt<Vec3b>(test, i, j) = Vec3b(0, 255, 0);
				}
			}
		imwrite("process/wetinwet-debug.jpg",test);
	}

	{//makeMask
		for (int i = 0; i < n; i++)
			for (int j = 0; j < m; j++) if (dataAt<uchar>(myCanny, i, j) != 0){
				if (!visit[i][j]){
					Debug() << "[ERROR] point not visit "<<i<<" "<<j;
					break;
				}
				myPoint p1(i, j), p2 = nearest[i][j];
				int dx = p1.x - p2.x, dy = p1.y - p2.y;
				if (dx != 0){
					double dd = double(dy) / dx;
					for (int k = p1.x; k <= p2.x; k++){
						int ddy = dd*(k - p1.x);
						if (ArraySpace::inMap(mask, k, ddy)){
							dataAt<uchar>(mask, k, ddy) = 1;
						}
					}
					for (int k = p2.x; k <= p1.x; k++){
						int ddy = dd*(k - p2.x);
						if (ArraySpace::inMap(mask, k, ddy)){
							dataAt<uchar>(mask, k, ddy) = 1;
						}
					}
				}
				else{
					for (int k = p1.y; k <= p2.y; k++){
						dataAt<uchar>(mask, p1.x, k) = 1;
					}
					for (int k = p2.y; k <= p1.y; k++){
						dataAt<uchar>(mask, p1.x, k) = 1;
					}
				}
			}
		//imwrite("process/wetinwet-maskline.jpg", mask * 255);
		for (int i = 0; i < n; i++)
			for (int j = 0; j < m;j++) 
				if (dataAt<uchar>(mask, i, j) != 0){
					for (int k = 0; k < 4; k++){
						int nx = i + ArraySpace::xo[k];
						int ny = j + ArraySpace::yo[k];
						if (ArraySpace::inMap(mask,myPoint( nx, ny))){
							dataAt<uchar>(mask, i, j) = 1;
						}
					}
				}
		
		for (int i = 0; i < n; i++)
			for (int j = 0; j < m; j++) if (dataAt<uchar>(mask, i, j) != 0 && dataAt<uchar>(myCanny, i, j) != 0){
				dataAt<uchar>(mask, i, j) = 0;
			}
		//imwrite("process/wetinwet-mask.jpg", mask * 255);
	}
	{//meanFilter
		for (int i = 0; i < n; i++)
			for (int j = 0; j < m; j++) if (dataAt<uchar>(mask, i, j) != 0){
				Scalar s = getMeanPoint<Vec3b>(dst, i, j, 5);
				dataAt<Vec3b>(dst, i, j) = Vec3b(s[0], s[1], s[2]);
			}
	}*/
}

#include <iostream>
#include <opencv2/opencv.hpp>
#include <netcdf.h>

using namespace cv;
using namespace std;

// resample.cpp
Mat resample_unsort(Mat &sind, Mat &img);
Mat resample_sort(Mat &sind, Mat &img);

// utils.cpp
void ncfatal(int n);
void fatal(string msg);

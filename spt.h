#include <iostream>
#include <opencv2/opencv.hpp>
#include <netcdf.h>

using namespace cv;
using namespace std;

#define SQ(x) ((x)*(x))

enum {
	MaskInvalid       = (1<<0),    		// or Valid
	MaskDay           = (1<<1),         // or Night
	MaskLand          = (1<<2),         // or Ocean
	MaskTwilightZone  = (1<<3),         // or No Twilight Zone
	MaskGlint         = (1<<4),         // or No Sun Glint
	MaskIce           = (1<<5),         // or No Ice
	
	MaskCloudOffset   = 6,              // first bit of cloud mask
	MaskCloud         = (1<<7)|(1<<6),
	MaskCloudClear    = 0,              // 0 - Clear
	MaskCloudProbably = (1<<6),         // 1 - Probably cloudy
	MaskCloudSure     = (1<<7),         // 2 - Confidently  cloudy
	MaskCloudInvalid  = (1<<7)|(1<<6),  // 3 - Irrelevant to SST (which includes land, ice and invalid pixels)
};

// resample.cpp
Mat resample_interp(Mat &slat, Mat &simg, Mat &slandmask);
Mat resample_float64(Mat &img, Mat &lat, Mat &acspo);

// utils.cpp
void ncfatal(int n);
void fatal(string msg);
string type2str(int type);
void checktype(Mat &mat, string name, int type);
void cmapimshow(string name, Mat &img, int cmap);

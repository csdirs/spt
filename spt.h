#include <opencv2/opencv.hpp>
#include <netcdf.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <omp.h>

#include "connectedcomponents.h"

using namespace cv;

#define SQ(x)    ((x)*(x))
#define SGN(A)   ((A) > 0 ? 1 : ((A) < 0 ? -1 : 0 ))
#define nelem(x) (sizeof(x)/sizeof((x)[0]))

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

enum {
	VIIRS_SWATH_SIZE = 16,
	MODIS_SWATH_SIZE = 10,
};

// resample.cpp
Mat resample_float32(Mat &img, Mat &lat, Mat &acspo, Mat &sind);
Mat resample_unsort(Mat &sind, Mat &img);
Mat resample_sort(Mat &sind, Mat &img);

// utils.cpp
void eprintf(const char *fmt, ...);
void logprintf(const char *fmt, ...);
char* estrdup(const char *s);
const char *type2str(int type);
void gray2rgb(Mat &src, Mat &dst, int cmap);
void cmapimshow(string name, Mat &img, int cmap);

// io.cpp
void savebin(const char *filename, Mat &m);
void savenpy(const char *filename, Mat &mat);
Mat readvar(int ncid, const char *name);
void ncfatal(int n, const char *fmt, ...);

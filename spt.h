//
// SST Pattern Test
//

#include <opencv2/opencv.hpp>
#include <netcdf.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <omp.h>
#include <unistd.h>	// access(3)

#include "connectedcomponents.h"

using namespace cv;

#define SQ(x)    ((x)*(x))
#define SGN(A)   ((A) > 0 ? 1 : ((A) < 0 ? -1 : 0 ))
#define nelem(x) (sizeof(x)/sizeof((x)[0]))
#define SAVENPY(X)	savenpy(#X ".npy", (X))
#define SAVENC(X)	if(DEBUG)savenc(#X ".nc", (X))
#define CHECKMAT(M, T)	CV_Assert((M).type() == (T) && (M).isContinuous())

#define GRAD_THRESH 0.2
#define GRAD_LOW 0.1
#define DELTARANGE_THRESH 0.3
#define DELTAMAG_LOW (GRAD_LOW/2.0)
#define LAM2_THRESH	-0.05
#define SST_LOW 270
#define SST_HIGH 309
#define DELTA_LOW -1
#define DELTA_HIGH 3
#define OMEGA_LOW -5
#define OMEGA_HIGH 0
#define ANOMALY_HIGH 10
#define ANOMALY_LOW -10
#define ANOMALY_THRESH -8
#define ALBEDO_LOW 3
#define ALBEDO_HIGH 4
#define EDGE_THRESH 1
#define STD_THRESH 0.5

#define TQ_STEP 1
#define DQ_STEP 0.2	// 0.5 doesn't work on some examples
#define OQ_STEP 0.5
#define AQ_STEP 1

#define TQ_HIST_STEP 1
#define DQ_HIST_STEP 0.25
#define OQ_HIST_STEP OQ_STEP

enum {
	DEBUG = 1,
	
	LUT_INVALID = -1,
	LUT_OCEAN = 0,
	LUT_CLOUD = 1,
	
	LUT_LAT_SPLIT = 4,
	
	COMP_INVALID = -1,	// invalid component
	COMP_SPECKLE = -2,	// component that is too small
};

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

typedef struct Resample Resample;
struct Resample {
	Mat sind, slat, sacspo, slandmask;
};

// resample.cc
Mat resample_unsort(const Mat &sind, const Mat &img);
Mat resample_sort(const Mat &sind, const Mat &img);
void resample_init(Resample *r, const Mat &lat, const Mat &acspo);
void resample_float32(const Resample *r, const Mat &src, Mat &dst, bool sort);

// utils.cc
void eprintf(const char *fmt, ...);
void logprintf(const char *fmt, ...);
char* estrdup(const char *s);
void *emalloc(size_t n);
const char *type2str(int type);
void diffcloudmask(const Mat &_old, const Mat &_new, Mat &_rgb);

// io.cc
char	*fileprefix(const char *path);
void	readvar(int ncid, const char *name, Mat&);
void ncfatal(int n, const char *fmt, ...);
int open_resampled(const char *path, Resample *r, int omode);
Mat	readvar_resampled(int ncid, Resample *r, const char *name);
void	savenc(const char *filename, const Mat &mat);
void loadnc(const char *filename, Mat &mat);

// npy.cc
void savenpy(const char *filename, Mat &mat);
void loadnpy(const char *filename, Mat &mat);

// filters.cc
void	laplacian(Mat &src, Mat &dst);
void	nanblur(const Mat &src, Mat &dst, int ksize);
void	gradientmag(const Mat &src, Mat &dst, Mat &dX, Mat &dY);
void	gradientmag(const Mat &src, Mat &dst);
void	localmax(const Mat &sstmag, Mat &high, Mat &low, int sigma);
void	stdfilter(const Mat &src, Mat &dst, int ksize);
void	rangefilter(const Mat &src, Mat &dst, int ksize);
void	logkernel(int n, double sigma, Mat &dst);
void	nanlogfilter(const Mat &src, const int size, const int sigma, const int factor, Mat &dst);

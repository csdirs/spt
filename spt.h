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
#define SAVENPY(X)	savenpy(#X ".npy", (X))

#define GRAD_THRESH 0.3
#define GRAD_LOW 0.1
#define SST_LOW 270
#define SST_HIGH 309
#define DELTA_LOW -1
#define DELTA_HIGH 3
#define OMEGA_LOW -5	// TODO: compute from data
#define OMEGA_HIGH 0	// TODO: compute from data
#define ALBEDO_LOW 3
#define ALBEDO_HIGH 4
#define EDGE_THRESH 1
#define STD_THRESH 0.5

#define TQ_STEP 3
#define DQ_STEP 0.5
#define OQ_STEP 0.5

#define TQ_HIST_STEP 1
#define DQ_HIST_STEP 0.25
#define OQ_HIST_STEP OQ_STEP

enum {
	DEBUG = 1,
	
	LUT_UNKNOWN = -1,
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

// resample.cpp
Mat resample_unsort(const Mat &sind, const Mat &img);
Mat resample_sort(const Mat &sind, const Mat &img);
void resample_init(Resample *r, const Mat &lat, const Mat &acspo);
void resample_float32(const Resample *r, const Mat &src, Mat &dst);

// utils.cpp
void eprintf(const char *fmt, ...);
void logprintf(const char *fmt, ...);
char* estrdup(const char *s);
void *emalloc(size_t n);
const char *type2str(int type);
void gray2rgb(Mat &src, Mat &dst, int cmap);
void cmapimshow(string name, Mat &img, int cmap);

// io.cpp
void savebin(const char *filename, Mat &m);
void savenpy(const char *filename, Mat &mat);
Mat readvar(int ncid, const char *name);
void ncfatal(int n, const char *fmt, ...);
int open_resampled(const char *path, Resample *r);
Mat	readvar_resampled(int ncid, Resample *r, const char *name);


// filters.cpp
void	laplacian(Mat &src, Mat &dst);
void	nanblur(const Mat &src, Mat &dst, int ksize);
void	gradientmag(const Mat &src, Mat &dst);
void	localmax(const Mat &gradmag, Mat &high, Mat &low, int sigma);
void	stdfilter(const Mat &src, Mat &dst, int ksize);

// quantize.cpp
void	quantize(const Mat &_lat, const Mat &_sst, const Mat &_delta, Mat &_omega,
	const Mat &_gradmag, Mat &_albedo, Mat &_acspo,
	Mat &TQ, Mat &DQ, Mat &OQ, Mat &lut);
int	quantize_lat(float lat);
int	quantize_sst(float sst);
int	quantize_delta(float delta);
int	quantize_omega(float omega);

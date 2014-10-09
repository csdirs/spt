#include "spt.h"

template <class T>
static Mat
resample_unsort_(Mat &sind, Mat &img)
{
	Mat newimg;
	int i, j, k;
	int32_t *sp;
	T *ip;

	CV_Assert(sind.type() == CV_32SC1);
	CV_Assert(img.channels() == 1);

	newimg = Mat::zeros(img.rows, img.cols, img.type());
	sp = (int32_t*)sind.data;
	ip = (T*)img.data;
	k = 0;
	for(i = 0; i < newimg.rows; i++){
		for(j = 0; j < newimg.cols; j++){
			newimg.at<T>(sp[k], j) = ip[k];
			k++;
		}
	}
	return newimg;
}

// Returns the unsorted image of the sorted image img.
// Sind is the image of sort indices.
Mat
resample_unsort(Mat &sind, Mat &img)
{
	switch(img.type()){
	default:
		eprintf("unsupported type %s\n", type2str(img.type()));
		break;
	case CV_8UC1:
		return resample_unsort_<uchar>(sind, img);
		break;
	case CV_32FC1:
		return resample_unsort_<float>(sind, img);
		break;
	case CV_64FC1:
		return resample_unsort_<double>(sind, img);
		break;
	}
	// not reached
	return Mat();
}

template <class T>
static Mat
resample_sort_(Mat &sind, Mat &img)
{
	Mat newimg;
	int i, j, k;
	int32_t *sp;
	T *np;

	CV_Assert(sind.type() == CV_32SC1);
	CV_Assert(img.channels() == 1);

	newimg = Mat::zeros(img.rows, img.cols, img.type());
	sp = (int*)sind.data;
	np = (T*)newimg.data;
	k = 0;
	for(i = 0; i < newimg.rows; i++){
		for(j = 0; j < newimg.cols; j++){
			np[k] = img.at<T>(sp[k], j);
			k++;
		}
	}
	return newimg;
}

// Returns the sorted image of the unsorted image img.
// Sind is the image of sort indices.
Mat
resample_sort(Mat &sind, Mat &img)
{
	switch(img.type()){
	default:
		eprintf("unsupported type %s\n", type2str(img.type()));
		break;
	case CV_8UC1:
		return resample_sort_<uchar>(sind, img);
		break;
	case CV_32FC1:
		return resample_sort_<float>(sind, img);
		break;
	case CV_64FC1:
		return resample_sort_<double>(sind, img);
		break;
	}
	// not reached
	return Mat();
}

// Returns the average of 3 pixels.
static double
avg3(double a, double b, double c)
{
	if(isnan(b))
		return NAN;
	if(isnan(a) || isnan(c))
		return b;
	return (a+b+c)/3.0;
}

// Returns the average filter of image 'in' with a window of 3x1
// where sorted order is not the same as the original order.
// Sind is the sort indices giving the sort order.
static Mat
avgfilter3(Mat &in, Mat &sind)
{
	Mat out;
	int i, j, rows, cols, *sindp;
	float *ip, *op;

	CV_Assert(in.type() == CV_32FC1);
	CV_Assert(sind.type() == CV_32SC1);
	rows = in.rows;
	cols = in.cols;

	out.create(rows, cols, CV_32FC1);
	in.row(0).copyTo(out.row(0));
	in.row(rows-1).copyTo(out.row(rows-1));

	for(i = 1; i < rows-1; i++){
		ip = in.ptr<float>(i);
		op = out.ptr<float>(i);
		sindp = sind.ptr<int>(i);
		for(j = 0; j < cols; j++){
			if(sindp[j] != i)
				op[j] = avg3(ip[j-cols], ip[j], ip[j+cols]);
			else
				op[j] = ip[j];
		}
	}
	return out;
}

// Interpolate the missing values in image simg and returns the result.
// Slat is the latitude image, and slandmask is the land mask image.
// All input arguments must already be sorted.
Mat
resample_interp(Mat &simg, Mat &slat, Mat &slandmask)
{
	int i, j, k, nbuf, *buf;
	Mat newimg, bufmat;
	double x, llat, rlat, lval, rval;

	CV_Assert(simg.type() == CV_32FC1);
	CV_Assert(slat.type() == CV_32FC1);
	CV_Assert(slandmask.type() == CV_8UC1);

	newimg = simg.clone();
	bufmat = Mat::zeros(simg.rows, 1, CV_32SC1);
	buf = (int*)bufmat.data;

	for(j = 0; j < simg.cols; j++){
		nbuf = 0;
		llat = -999;
		lval = NAN;
		for(i = 0; i < simg.rows; i++){
			// land pixel, nothing to do
			if(slandmask.at<unsigned char>(i, j) != 0){
				continue;
			}

			// valid pixel
			if(!isnan(simg.at<float>(i, j))){
				// first pixel is not valid, so extrapolate
				if(llat == -999){
					for(k = 0; k < nbuf; k++){
						newimg.at<float>(buf[k],j) = simg.at<float>(i, j);
					}
					nbuf = 0;
				}

				// interpolate pixels in buffer
				for(k = 0; k < nbuf; k++){
					rlat = slat.at<float>(i, j);
					rval = simg.at<float>(i, j);
					x = slat.at<float>(buf[k], j);
					newimg.at<float>(buf[k],j) =
						lval + (rval - lval)*(x - llat)/(rlat - llat);
				}

				llat = slat.at<float>(i, j);
				lval = simg.at<float>(i, j);
				nbuf = 0;
				continue;
			}

			// not land and no valid pixel
			buf[nbuf++] = i;
		}
		// extrapolate the last pixels
		if(llat != -999){
			for(k = 0; k < nbuf; k++){
				newimg.at<float>(buf[k],j) = lval;
			}
		}
	}
	return newimg;
}

enum Pole {
	NORTHPOLE,
	SOUTHPOLE,
	NOPOLE,
};
typedef enum Pole Pole;

// Argsort latitude image 'lat' with given swath size.
// Image of sort indices are return in 'sortidx'.
static void
argsortlat(Mat &lat, int swathsize, Mat &sortidx)
{
	int i, j, off, width, height, dir, d, split;
	Pole pole;
	Mat col, idx, botidx;
	Range colrg, toprg, botrg;
	
	CV_Assert(lat.type() == CV_32FC1);
	CV_Assert(swathsize >= 2);
	CV_Assert(lat.data != sortidx.data);
	
	width = lat.cols;
	height = lat.rows;
	sortidx.create(height, width, CV_32SC1);
	
	// For a column in latitude image, look at every 'swathsize' pixels
	// starting from 'off'. If they increases and then decreases, or
	// decreases and then increases, we're at the polar region.
	off = swathsize/2;
	
	pole = NOPOLE;
	
	for(j = 0; j < width; j++){
		col = lat.col(j);
		
		// find initial direction -- increase, decrease or no change
		dir = 0;
		for(i = off+swathsize; i < height; i += swathsize){
			dir = SGN(col.at<float>(i) - col.at<float>(i-swathsize));
			if(dir != 0)
				break;
		}
		
		// find change in direction if there is one
		for(; i < height; i += swathsize){
			d = SGN(col.at<float>(i) - col.at<float>(i-swathsize));
			if(dir == 1 && d == -1){
				CV_Assert(pole == NOPOLE || pole == NORTHPOLE);
				pole = NORTHPOLE;
				break;
			}
			if(dir == -1 && d == 1){
				CV_Assert(pole == NOPOLE || pole == SOUTHPOLE);
				pole = SOUTHPOLE;
				break;
			}
		}
		
		if(i >= height){
			pole = NOPOLE;
			if(dir >= 0)
				sortIdx(col, sortidx.col(j), CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);
			else
				sortIdx(col, sortidx.col(j), CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING);
			continue;
		}
		
		split = i-swathsize;	// split before change in direction
		colrg = Range(j, j+1);
		toprg = Range(0, split);
		botrg = Range(split, height);
		
		if(pole == NORTHPOLE){
			botidx = sortidx(botrg, colrg);
			sortIdx(col.rowRange(toprg), sortidx(toprg, colrg),
				CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);
			sortIdx(col.rowRange(botrg), botidx,
				CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING);
			botidx += split;
		}else{	// pole == SOUTHPOLE
			botidx = sortidx(botrg, colrg);
			sortIdx(col.rowRange(toprg), sortidx(toprg, colrg),
				CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING);
			sortIdx(col.rowRange(botrg), botidx,
				CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);
			botidx += split;
		}
	}
}

// Resample VIIRS swatch image img with corresponding
// latitude image lat, and ACSPO mask acspo.
Mat
resample_float32(Mat &img, Mat &lat, Mat &acspo)
{
	Mat sind, landmask;

	CV_Assert(img.type() == CV_32FC1);
	CV_Assert(lat.type() == CV_32FC1);
	CV_Assert(acspo.type() == CV_8UC1);

	argsortlat(lat, VIIRS_SWATH_SIZE, sind);

	img = resample_sort(sind, img);
	img = avgfilter3(img, sind);

	lat = resample_sort(sind, lat);
	acspo = resample_sort(sind, acspo);
	landmask = (acspo & MaskLand) != 0;

	return resample_interp(img, lat, landmask);
}

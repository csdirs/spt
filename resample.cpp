#include "spt.h"

Mat
resample_unsort(Mat &sind, Mat &img)
{
	Mat newimg;
	int i, j, k;
	int32_t *sp;
	double *ip;

	CV_Assert(sind.type() == CV_32SC1);
	CV_Assert(img.type() == CV_64FC1);

	newimg = Mat::zeros(img.rows, img.cols, CV_64F);
	sp = (int32_t*)sind.data;
	ip = (double*)img.data;
	k = 0;
	for(i = 0; i < newimg.rows; i++){
		for(j = 0; j < newimg.cols; j++){
			newimg.at<double>(sp[k], j) = ip[k];
			k++;
		}
	}
	return newimg;
}

template <class T>
Mat
resample_sort(Mat &sind, Mat &img, int type)
{
	Mat newimg;
	int i, j, k;
	int32_t *sp;
	T *np;

	CV_Assert(sind.type() == CV_32SC1);

	newimg = Mat::zeros(img.rows, img.cols, type);
	sp = (int32_t*)sind.data;
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

static double
avg3(double a, double b, double c)
{
	if(isnan(b))
		return NAN;
	if(isnan(a) || isnan(c))
		return b;
	return (a+b+c)/3.0;
}

// Average filter of image 'in' with a window of 3x1
// where sorted order is not the same as the origial order.
static void
avgfilter3(Mat &in, Mat &out, Mat &sind)
{
	int i, j, rows, cols, *sindp;
	double *ip, *op;

	CV_Assert(in.type() == CV_64FC1);
	CV_Assert(sind.type() == CV_32SC1);
	rows = in.rows;
	cols = in.cols;

	out.create(rows, cols, CV_64FC1);
	in.row(0).copyTo(out.row(0));
	in.row(rows-1).copyTo(out.row(rows-1));

	for(i = 1; i < rows-1; i++){
		ip = in.ptr<double>(i);
		op = out.ptr<double>(i);
		sindp = sind.ptr<int>(i);
		for(j = 0; j < cols; j++){
			if(sindp[j] != i)
				op[j] = avg3(ip[j-cols], ip[j], ip[j+cols]);
			else
				op[j] = ip[j];
		}
	}
}

Mat
resample_interp(Mat &simg, Mat &slat, Mat &slandmask)
{
	int i, j, k, nbuf, *buf;
	Mat newimg, bufmat;
	double x, llat, rlat, lval, rval;

	CV_Assert(simg.type() == CV_64FC1);
	CV_Assert(slat.type() == CV_64FC1);
	CV_Assert(slandmask.type() == CV_8UC1);

	newimg = simg.clone();
	bufmat = Mat::zeros(simg.rows, 1, CV_32SC1);
	buf = (int*)bufmat.data;

	for(j = 0; j < simg.cols; j++){
		nbuf = 0;
		llat = -999;
		for(i = 0; i < simg.rows; i++){
			// land pixel, nothing to do
			if(slandmask.at<unsigned char>(i, j) != 0){
				continue;
			}

			// valid pixel
			if(!isnan(simg.at<double>(i, j))){
				// first pixel is not valid, so extrapolate
				if(llat == -999){
					for(k = 0; k < nbuf; k++){
						newimg.at<double>(buf[k],j) = simg.at<double>(i, j);
					}
					nbuf = 0;
				}

				// interpolate pixels in buffer
				for(k = 0; k < nbuf; k++){
					rlat = slat.at<double>(i, j);
					rval = simg.at<double>(i, j);
					x = slat.at<double>(buf[k], j);
					newimg.at<double>(buf[k],j) =
						lval + (rval - lval)*(x - llat)/(rlat - llat);
				}

				llat = slat.at<double>(i, j);
				lval = simg.at<double>(i, j);
				nbuf = 0;
				continue;
			}

			// not land and no valid pixel
			buf[nbuf++] = i;
		}
		// extrapolate the last pixels
		if(llat != -999){
			for(k = 0; k < nbuf; k++){
				newimg.at<double>(buf[k],j) = lval;
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
	int i, j, off, width, height, dir, d;
	Pole pole;
	Mat col, idx, A, B;
	Range colrg, toprg, botrg;
	
	CV_Assert(lat.type() == CV_64FC1);
	CV_Assert(swathsize >= 2);
	
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
		for(i = off; i < height; i += swathsize){
			dir = SGN(col.at<double>(i+swathsize) - col.at<double>(i));
			if(dir != 0)
				break;
		}
		
		// find change in direction if there is one
		for(; i < height; i += swathsize){
			d = SGN(col.at<double>(i+swathsize) - col.at<double>(i));
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
		if(i >= height)
			pole = NOPOLE;
//printf("j = %4d, pole = %d\n", j, pole);
		colrg = Range(j, j+1);
		toprg = Range(0, i);
		botrg = Range(i, height);

		// argsort latitudes for this column
		switch(pole){
		case NOPOLE:
			if(dir >= 0)
				sortIdx(col, sortidx.col(j), CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);
			else
				sortIdx(col, sortidx.col(j), CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING);
			break;
		case NORTHPOLE:
			sortIdx(col.rowRange(toprg), sortidx(toprg, colrg),
				CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);
			sortIdx(col.rowRange(botrg), sortidx(botrg, colrg),
				CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING);
			break;
		case SOUTHPOLE:
			sortIdx(col.rowRange(toprg), sortidx(toprg, colrg),
				CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING);
			sortIdx(col.rowRange(botrg), sortidx(botrg, colrg),
				CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);
			break;
		}
	}
}

Mat
resample_float64(Mat &img, Mat &lat, Mat &acspo)
{
	Mat sind, landmask, tmpmat;

	CV_Assert(img.type() == CV_64FC1);
	CV_Assert(lat.type() == CV_64FC1);
	CV_Assert(acspo.type() == CV_8UC1);

	//sortIdx(lat, sind, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);
	argsortlat(lat, 10, sind);
//dumpmat("sortind.bin", sind);

	img = resample_sort<double>(sind, img, CV_64FC1);
//dumpmat("sortsst.bin", img);
	avgfilter3(img, tmpmat, sind);
	img = tmpmat;
//dumpmat("medfiltsst.bin", img);

	lat = resample_sort<double>(sind, lat, CV_64FC1);
	acspo = resample_sort<unsigned char>(sind, acspo, CV_8UC1);
	landmask = (acspo & MaskLand) != 0;

	return resample_interp(img, lat, landmask);
}

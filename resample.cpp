#include "spt.h"

Mat
resample_unsort(Mat &sind, Mat &img)
{
	Mat newimg;
	int i, j, k;
	int32_t *sp;
	double *ip;

	checktype(sind, "resample_unsort:sind", CV_32SC1);
	checktype(img, "resample_unsort:img", CV_64FC1);

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

	checktype(sind, "resample_sort:sind", CV_32SC1);

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

#define ISNAN(x) ((x) != (x))

// TODO: median filter?
Mat
resample_interp(Mat &simg, Mat &slat, Mat &slandmask)
{
	int i, j, k, nbuf, *buf;
	Mat newimg, bufmat;
	double x, llat, rlat, lval, rval;

	checktype(simg, "resample_interp:simg", CV_64FC1);
	checktype(slat, "resample_interp:slat", CV_64FC1);
	checktype(slandmask, "resample_interp:slandmask", CV_8UC1);

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
			if(!ISNAN(simg.at<double>(i, j))){
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

Mat
resample_float64(Mat &img, Mat &lat, Mat &acspo)
{
	Mat sind, landmask;

	checktype(img, "resample_float64:img", CV_64FC1);
	checktype(lat, "resample_float64:lat", CV_64FC1);
	checktype(acspo, "resample_float64:acspo", CV_8UC1);

	sortIdx(lat, sind, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);
	img = resample_sort<double>(sind, img, CV_64FC1);
	lat = resample_sort<double>(sind, lat, CV_64FC1);
	acspo = resample_sort<unsigned char>(sind, acspo, CV_8UC1);
	landmask = (acspo & MaskLand) != 0;

	return resample_interp(img, lat, landmask);
}

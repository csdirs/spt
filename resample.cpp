#include "spt.h"

Mat
resample_unsort(Mat &sind, Mat &img)
{
	Mat newimg;
	int i, j, k;
	int32_t *sp;
	float *ip;

	if(img.type() != CV_32FC1){
		fatal("invalid img type");
	}
	newimg = Mat::zeros(img.rows, img.cols, CV_64F);
	sp = (int32_t*)sind.data;
	ip = (float*)img.data;
	k = 0;
	for(i = 0; i < newimg.rows; i++){
		for(j = 0; j < newimg.cols; j++){
			newimg.at<double>(sp[k], j) = ip[k];
			k++;
		}
	}
	return newimg;
}

Mat
resample_sort(Mat &sind, Mat &img)
{
	Mat newimg;
	int i, j, k;
	int32_t *sp;
	double *np;

	if(img.type() != CV_32FC1){
		fatal("invalid img type");
	}
	newimg = Mat::zeros(img.rows, img.cols, CV_64F);
	sp = (int32_t*)sind.data;
	np = (double*)newimg.data;
	k = 0;
	for(i = 0; i < newimg.rows; i++){
		for(j = 0; j < newimg.cols; j++){
			np[k] = img.at<float>(sp[k], j);
			k++;
		}
	}
	return newimg;
}

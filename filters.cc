//
// Image processing filters
//

#include "spt.h"

// Compute the laplacian filter of image src into dst.
//
// src -- source image
// dst -- destination image (output)
//
void
laplacian(const Mat &src, Mat &dst)
{
	 Mat kern = (Mat_<double>(3,3) <<
	 	0,     1/4.0,  0,
		1/4.0, -4/4.0, 1/4.0,
		0,     1/4.0,  0);
	filter2D(src, dst, -1, kern);
}

// Separable blur implementation that can handle images containing NaN.
// OpenCV blur does not correctly handle such images.
//
// src -- source image
// dst -- destination image (output)
// ksize -- kernel size
//
void
nanblur(const Mat &src, Mat &dst, int ksize)
{
	Mat kernel = Mat::ones(ksize, 1, CV_64FC1)/ksize;
	sepFilter2D(src, dst, -1, kernel, kernel);
}

// Compute the gradient magnitude of image src into dst.
//
// src -- source image
// dX, dY -- derivative in the X and Y directions (output)
// dst -- destination image of gradient magnitude (output)
//
void
gradientmag(const Mat &src, Mat &dX, Mat &dY, Mat &dst)
{
	Mat h = (Mat_<double>(5,1) <<
		0.036420, 0.248972, 0.429217, 0.248972, 0.036420);
	Mat hp = (Mat_<double>(5,1) <<
		0.108415, 0.280353, 0, -0.280353, -0.108415);

	sepFilter2D(src, dX, -1, h, hp);
	// We negate h here to fix the sign of Dy
	sepFilter2D(src, dY, -1, hp, -h);
	sqrt(dX.mul(dX) + dY.mul(dY), dst);
}

//  Find local maximum.
//
// gradmag -- SST gradient magnitude
// high, low -- (output)
// sigma -- standard deviation
//
void
localmax(const Mat &gradmag, Mat &high, Mat &low, int sigma)
{
	enum {
		NStd = 3,
	};
	int i, j, x, y, winsz;
	double e, a, dd, mu1, mu2;
	float *Dxxp, *Dxyp, *Dyyp, *highp, *lowp;
	Mat DGaussxx, DGaussxy, DGaussyy,
		Dxx, Dxy, Dyy;
	
	CHECKMAT(gradmag, CV_32FC1);

	winsz = 2*(NStd*sigma) + 1;
	DGaussxx = Mat::zeros(winsz, winsz, CV_64FC1);
	DGaussxy = Mat::zeros(winsz, winsz, CV_64FC1);
	DGaussyy = Mat::zeros(winsz, winsz, CV_64FC1);

	for(i = 0; i < winsz; i++){
		x = i - NStd*sigma;
		for(j = 0; j < winsz; j++){
			y = j - NStd*sigma;

			e = exp(-(SQ(x) + SQ(y)) / (2*SQ(sigma)));
			DGaussxx.at<double>(i, j) =
				DGaussyy.at<double>(j, i) =
				1/(2*M_PI*pow(sigma, 4)) *
				(SQ(x)/SQ(sigma) - 1) * e;
			DGaussxy.at<double>(i, j) =
				1/(2*M_PI*pow(sigma, 6)) * (x*y) * e;
		}
	}
	filter2D(gradmag, Dxx, -1, DGaussxx);
	filter2D(gradmag, Dxy, -1, DGaussxy);
	filter2D(gradmag, Dyy, -1, DGaussyy);

	CHECKMAT(Dxx, CV_32FC1);
	CHECKMAT(Dxy, CV_32FC1);
	CHECKMAT(Dyy, CV_32FC1);
	
	high.create(Dxx.rows, Dxx.cols, CV_32FC1);
	low.create(Dxx.rows, Dxx.cols, CV_32FC1);
	highp = (float*)high.data;
	lowp = (float*)low.data;
	Dxxp = (float*)Dxx.data;
	Dxyp = (float*)Dxy.data;
	Dyyp = (float*)Dyy.data;
	for(i = 0; i < Dxx.rows*Dxx.cols; i++){
		a = Dxxp[i] + Dyyp[i];
		dd = sqrt(SQ(Dxxp[i] - Dyyp[i]) + 4*SQ(Dxyp[i]));
		mu1 = 0.5*(a + dd);
		mu2 = 0.5*(a - dd);
		if(abs(mu1) > abs(mu2)){
			highp[i] = mu1;
			lowp[i] = mu2;
		}else{
			highp[i] = mu2;
			lowp[i] = mu1;
		}
	}
}

// Standard deviation filter, implemented as
//	dst = sqrt(blur(src^2) - blur(src)^2)
//
// src -- source image
// dst -- destination image (output)
// ksize -- kernel size
//
void
stdfilter(const Mat &src, Mat &dst, int ksize)
{
	int i;
	Mat b, bs, _tmp;
	float *tmp;
	
	nanblur(src.mul(src), bs, ksize);
	nanblur(src, b, ksize);

	// avoid sqrt of nagative number
	_tmp = bs - b.mul(b);
	CHECKMAT(_tmp, CV_32FC1);
	tmp = (float*)_tmp.data;
	for(i = 0; i < (int)_tmp.total(); i++){
		if(tmp[i] < 0)
			tmp[i] = 0;
	}
	sqrt(_tmp, dst);
}

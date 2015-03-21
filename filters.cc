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
// dst -- destination image of gradient magnitude (output)
// dX, dY -- derivative in the X and Y directions (output)
//
void
gradientmag(const Mat &src, Mat &dst, Mat &dX, Mat &dY)
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

// Compute the gradient magnitude of image src into dst.
//
// src -- source image
// dst -- destination image of gradient magnitude (output)
//
void
gradientmag(const Mat &src, Mat &dst)
{
	Mat dX, dY;
	gradientmag(src, dst, dX, dY);
}

//  Find local maximum.
//
// sstmag -- SST gradient magnitude
// high, low -- (output)
// sigma -- standard deviation
//
void
localmax(const Mat &sstmag, Mat &high, Mat &low, int sigma)
{
	enum {
		NStd = 3,
	};
	int i, j, x, y, winsz;
	double e, a, dd, mu1, mu2;
	float *Dxxp, *Dxyp, *Dyyp, *highp, *lowp;
	Mat DGaussxx, DGaussxy, DGaussyy,
		Dxx, Dxy, Dyy;
	
	CHECKMAT(sstmag, CV_32FC1);

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
	filter2D(sstmag, Dxx, -1, DGaussxx);
	filter2D(sstmag, Dxy, -1, DGaussxy);
	filter2D(sstmag, Dyy, -1, DGaussyy);

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

// Range filter.
//
// src -- source image
// dst -- destination image (output)
// ksize -- kernel size
//
void
rangefilter(const Mat &src, Mat &dst, int ksize)
{
	Mat min, max;
	
	Mat elem = getStructuringElement(MORPH_RECT, Size(ksize, ksize));
	erode(src, min, elem);
	dilate(src, max, elem);
	dst = max - min;
}

// Compute Laplacian of Gaussian kernel.
// See http://homepages.inf.ed.ac.uk/rbf/HIPR2/log.htm
// This function is equivalent to fspecial('log', n, sigma) in MATLAB.
//
// n -- width/height of kernel
// sigma -- standard deviation of the Gaussian
// dst -- the kernel (output)
//
void
logkernel(int n, double sigma, Mat &dst)
{
	dst.create(n, n, CV_64FC1);
	int h = n/2;
	double ss = sigma*sigma;
	
	for(int i = 0; i < n; i++){
		double y = i - h;
		for(int j = 0; j < n; j++){
			double x = j - h;
			dst.at<double>(i, j) = exp(-(x*x + y*y) / (2*ss));
		}
	}
	double total = sum(dst)[0];
	
	for(int i = 0; i < n; i++){
		double y = i - h;
		for(int j = 0; j < n; j++){
			double x = j - h;
			dst.at<double>(i, j) *= x*x + y*y - 2*ss;
		}
	}

	dst /= pow(ss, 2) * total;
	dst -= mean(dst)[0];	// make sum of filter equal to 0
}

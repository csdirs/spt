#include "spt.h"

//#define DATAPATH "/cephfs/fhs/data/in/acspo/www.star.nesdis.noaa.gov/pub/sod/sst/micros_data/acspo_nc/npp/2014-07-10/ACSPO_V2.30_NPP_VIIRS_2014-07-10_1230-1240_20140713.061812.nc"
#define DATAPATH "/cephfs/fhs/data/in/acspo/www.star.nesdis.noaa.gov/pub/sod/osb/ykihai/VIIRS_Samples_for_Irina/Select/ACSPO_V2.30_NPP_VIIRS_2014-06-20_1710-1719_20140623.071032.nc"

Mat
readvar(int ncid, const char *name)
{
	int i, varid, n, dimids[2], cvt;
	size_t shape[2];
	nc_type nct;
	Mat img;
	
	if(n = nc_open(DATAPATH, NC_NOWRITE, &ncid))
		ncfatal(n);
	if(n = nc_inq_varid(ncid, name, &varid))
		ncfatal(n);
	if(n = nc_inq_var(ncid, varid, NULL, &nct, NULL, dimids, NULL))
		ncfatal(n);
	for(i = 0; i < 2; i++){
		if(n = nc_inq_dimlen(ncid, dimids[i], &shape[i]))
			ncfatal(n);
	}
	switch(nct){
	default:
		fatal("unknown netcdf data type");
		break;
	case NC_BYTE:
		img = Mat::zeros(shape[0], shape[1], CV_8SC1);
		if(n = nc_get_var_schar(ncid, varid, (signed char*)img.data))
			ncfatal(n);
		break;
	case NC_UBYTE:
		img = Mat::zeros(shape[0], shape[1], CV_8UC1);
		if(n = nc_get_var_uchar(ncid, varid, (unsigned char*)img.data))
			ncfatal(n);
		break;
	case NC_FLOAT:
		img = Mat::zeros(shape[0], shape[1], CV_32FC1);
		if(n = nc_get_var_float(ncid, varid, (float*)img.data))
			ncfatal(n);
		img.convertTo(img, CV_64F);
		break;
	}
	return img;
}

void
clipsst(Mat &sst)
{
	float *p;
	int i;

	p = (float*)sst.data;
	for(i = 0; i < sst.total(); i++){
		if(p[i] > 1000 || p[i] < -1000)
			p[i] = NAN;
	}
}

void
laplacian(Mat &src, Mat &dst)
{
	 Mat kern = (Mat_<double>(3,3) <<
	 	0,     1/4.0,  0,
		1/4.0, -4/4.0, 1/4.0,
		0,     1/4.0,  0);
	filter2D(src, dst, -1, kern);
}

void
gradientmag(Mat &img, Mat &gm)
{
    Mat h = (Mat_<double>(5,1) <<
		0.036420, 0.248972, 0.429217, 0.248972, 0.036420);
    Mat hp = (Mat_<double>(5,1) <<
		0.108415, 0.280353, 0, -0.280353, -0.108415);
	Mat tmp, dX, dY, ht, hpt;

	// TODO: padding needed here?
	transpose(h, ht);
	transpose(hp, hpt);
	filter2D(img, tmp, -1, hp);
	filter2D(tmp, dX, -1, ht);
	filter2D(img, tmp, -1, h);
	filter2D(tmp, dY, -1, hpt);
	cout << "dX rows/cols " << dX.rows << " " << dX.cols << endl;
    sqrt(dX.mul(dX) + dY.mul(dY), gm);
}

enum {
	NStd = 3,
};

void
localmax(Mat &gradmag, Mat &high, Mat &low, int sigma)
{
	int i, j, x, y, winsz;
	double e, a, dd, mu1, mu2, *Dxxp, *Dxyp, *Dyyp, *highp, *lowp;
	Mat DGaussxx, DGaussxy, DGaussyy,
		Dxx, Dxy, Dyy;

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

	high.create(Dxx.rows, Dxx.cols, CV_64FC1);
	low.create(Dxx.rows, Dxx.cols, CV_64FC1);
	highp = (double*)high.data;
	lowp = (double*)low.data;
	Dxxp = (double*)Dxx.data;
	Dxyp = (double*)Dxy.data;
	Dyyp = (double*)Dyy.data;
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

int
main(int argc, char **argv)
{
	Mat sst, lat, elem, sstdil, sstero, rfilt, sstlap, sind;
	Mat acspo, landmask, interpsst, gradmag, high, low;
	float *p, *q;
	int i, ncid, n;

	if(n = nc_open(DATAPATH, NC_NOWRITE, &ncid))
		ncfatal(n);
	sst = readvar(ncid, "sst_regression");
	lat = readvar(ncid, "latitude");
	acspo = readvar(ncid, "acspo_mask");
	if(n = nc_close(ncid))
		ncfatal(n);

	interpsst = resample_float64(sst, lat, acspo);

	elem = getStructuringElement(MORPH_RECT, Size(3, 3), Point(1, 1));
	dilate(interpsst, sstdil, elem);
	erode(interpsst, sstero, elem);
	subtract(sstdil, sstero, rfilt);

	laplacian(interpsst, sstlap);

	gradientmag(interpsst, gradmag);
	localmax(gradmag, high, low, 1);
	clipsst(rfilt);

	cmapimshow("SST", sst, COLORMAP_JET);
	cmapimshow("Rangefilt SST", rfilt, COLORMAP_JET);
	cmapimshow("Laplacian SST", sstlap, COLORMAP_JET);
	cmapimshow("acspo", acspo, COLORMAP_JET);
	cmapimshow("interpsst", interpsst, COLORMAP_JET);
	cmapimshow("gradmag", gradmag, COLORMAP_JET);
	cmapimshow("high", high, COLORMAP_JET);
	cmapimshow("low", low, COLORMAP_JET);
	//namedWindow("SortIdx SST", CV_WINDOW_NORMAL|CV_WINDOW_KEEPRATIO);
	//imshow("SortIdx SST", sind);
	while(waitKey(0) != 'q')
		;

	return 0;
}

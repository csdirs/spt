#include <iostream>
#include <opencv2/opencv.hpp>
#include <netcdf.h>

using namespace cv;
using namespace std;

#define DATAPATH "/cephfs/fhs/data/in/acspo/www.star.nesdis.noaa.gov/pub/sod/sst/micros_data/acspo_nc/npp/2014-07-10/ACSPO_V2.30_NPP_VIIRS_2014-07-10_1230-1240_20140713.061812.nc"

void
ncfatal(int n)
{
	cerr << "Error: " << nc_strerror(n) << endl;
	exit(2);
}

Mat
readsst(void)
{
	int i, ncid, varid, n, dimids[2];
	size_t shape[2];
	Mat sst;
	
	if(n = nc_open(DATAPATH, NC_NOWRITE, &ncid))
		ncfatal(n);
	if(n = nc_inq_varid(ncid, "sst_regression", &varid))
		ncfatal(n);
	if(n = nc_inq_vardimid(ncid, varid, dimids))
		ncfatal(n);
	for(i = 0; i < 2; i++){
		if(n = nc_inq_dimlen(ncid, dimids[i], &shape[i]))
			ncfatal(n);
	}
	sst = Mat::zeros(shape[0], shape[1], CV_32FC1);
	if(n = nc_get_var_float(ncid, varid, (float*)sst.data))
		ncfatal(n);
	if(n = nc_close(ncid))
		ncfatal(n);

	return sst;
}

void
cmapImshow(Mat &img, int cmap, double scale)
{
	int i, key;
	double min, max;
	float *p;
	Mat tmp, tmp2, image;

	minMaxLoc(img, &min, &max, NULL, NULL);
	i = 0;
	p = (float*)img.data;
	for(i = 0; i < img.total(); i++){
		p[i] -= min;
		p[i] /= max-min;
		p[i] *= 255;
	}

	img.convertTo(tmp, CV_8U);
	resize(tmp, tmp2, Size(), scale, scale);
	applyColorMap(tmp2, image, cmap);

	namedWindow("Display window", WINDOW_AUTOSIZE);
	imshow("Display window", image);
	do{
		key = waitKey(0);
	}while(key != 'q');
}

int
main(int argc, char **argv)
{
	Mat sst, tmp, tmp2, image;

	sst = readsst();
	if(sst.empty()){	// Check for invalid input
		cout <<  "could not read data" << std::endl ;
		return -1;
	}
	cmapImshow(sst, COLORMAP_JET, 0.2);

	return 0;
}

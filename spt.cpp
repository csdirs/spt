#include "spt.h"

#define DATAPATH "/cephfs/fhs/data/in/acspo/www.star.nesdis.noaa.gov/pub/sod/sst/micros_data/acspo_nc/npp/2014-07-10/ACSPO_V2.30_NPP_VIIRS_2014-07-10_1230-1240_20140713.061812.nc"

Mat
readvar(int ncid, const char *name)
{
	int i, varid, n, dimids[2];
	size_t shape[2];
	Mat img;
	
	if(n = nc_open(DATAPATH, NC_NOWRITE, &ncid))
		ncfatal(n);
	if(n = nc_inq_varid(ncid, name, &varid))
		ncfatal(n);
	if(n = nc_inq_vardimid(ncid, varid, dimids))
		ncfatal(n);
	for(i = 0; i < 2; i++){
		if(n = nc_inq_dimlen(ncid, dimids[i], &shape[i]))
			ncfatal(n);
	}
	img = Mat::zeros(shape[0], shape[1], CV_32FC1);
	if(n = nc_get_var_float(ncid, varid, (float*)img.data))
		ncfatal(n);
	return img;
}

void
cmapImshow(string name, Mat &img, int cmap, double scale)
{
	double min, max;

	minMaxLoc(img, &min, &max, NULL, NULL);
	cout << name << " min: " << min << " max: " << max << endl;
	cout << name << " type: " << img.type() << endl;

	switch(img.type()){
	case CV_16SC1:
	case CV_32SC1:
		img.convertTo(img, CV_64F);
		// fallthrough
	case CV_32FC1:
	case CV_64FC1:
		img -= min;
		img.convertTo(img, CV_8U, 255.0/(max-min));
		break;
	}
	//resize(img, tmp2, Size(), scale, scale);
	applyColorMap(img, img, cmap);

	//namedWindow(name, WINDOW_AUTOSIZE);
	namedWindow(name, CV_WINDOW_NORMAL|CV_WINDOW_KEEPRATIO);
	imshow(name, img);
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

int
main(int argc, char **argv)
{
	Mat sst, lat, elem, sstdil, sstero, rfilt, sstlap, sind;
	float *p, *q;
	int i, ncid, n;

	if(n = nc_open(DATAPATH, NC_NOWRITE, &ncid))
		ncfatal(n);
	sst = readvar(ncid, "sst_regression");
	lat = readvar(ncid, "latitude");
	if(n = nc_close(ncid))
		ncfatal(n);

	if(sst.empty()){	// Check for invalid input
		cout <<  "could not read data" << std::endl ;
		return -1;
	}

	sortIdx(lat, sind, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);
	sst = resample_sort(sind, sst);

	elem = getStructuringElement(MORPH_RECT, Size(3, 3), Point(1, 1));
	dilate(sst, sstdil, elem);
	erode(sst, sstero, elem);
	subtract(sstdil, sstero, rfilt);

	laplacian(sst, sstlap);

	clipsst(rfilt);

	cmapImshow("SST", sst, COLORMAP_JET, 0.2);
	cmapImshow("Rangefilt SST", rfilt, COLORMAP_JET, 0.2);
	cmapImshow("Laplacian SST", sstlap, COLORMAP_JET, 0.2);
	cmapImshow("SortIdx Lat", sind, COLORMAP_JET, 0.2);
	//namedWindow("SortIdx SST", CV_WINDOW_NORMAL|CV_WINDOW_KEEPRATIO);
	//imshow("SortIdx SST", sind);
	while(waitKey(0) != 'q')
		;

	return 0;
}

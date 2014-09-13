#include "spt.h"

#define DATAPATH "/cephfs/fhs/data/in/acspo/www.star.nesdis.noaa.gov/pub/sod/sst/micros_data/acspo_nc/npp/2014-07-10/ACSPO_V2.30_NPP_VIIRS_2014-07-10_1230-1240_20140713.061812.nc"

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

int
main(int argc, char **argv)
{
	Mat sst, lat, elem, sstdil, sstero, rfilt, sstlap, sind;
	Mat acspo, landmask, interpsst;
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
	dilate(sst, sstdil, elem);
	erode(sst, sstero, elem);
	subtract(sstdil, sstero, rfilt);

	laplacian(sst, sstlap);

	clipsst(rfilt);

	cmapImshow("SST", sst, COLORMAP_JET);
	cmapImshow("Rangefilt SST", rfilt, COLORMAP_JET);
	cmapImshow("Laplacian SST", sstlap, COLORMAP_JET);
	cmapImshow("acspo", acspo, COLORMAP_JET);
	cmapImshow("interpsst", interpsst, COLORMAP_JET);
	//namedWindow("SortIdx SST", CV_WINDOW_NORMAL|CV_WINDOW_KEEPRATIO);
	//imshow("SortIdx SST", sind);
	while(waitKey(0) != 'q')
		;

	return 0;
}

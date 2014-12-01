#include "spt.h"

void
granulelut(const char *path, Mat &lut)
{
	Mat sst, lat, lon, m14, m15, m16, medf, stdf, blurf;
	Mat acspo, gradmag, delta, omega, albedo, TQ, DQ, OQ, lam1, lam2, easyclouds, easyfronts;
	int ncid, n;
	Resample *r;

	logprintf("reading and resampling...\n");
	r = new Resample;
	ncid = open_resampled(path, r);
	sst = readvar_resampled(ncid, r, "sst_regression");
	lat = readvar_resampled(ncid, r, "latitude");
	lon = readvar_resampled(ncid, r, "longitude");
	// TODO: interpolate acspo
	acspo = readvar_resampled(ncid, r, "acspo_mask");
	m14 = readvar_resampled(ncid, r, "brightness_temp_chM14");
	m15 = readvar_resampled(ncid, r, "brightness_temp_chM15");
	m16 = readvar_resampled(ncid, r, "brightness_temp_chM16");
	albedo = readvar_resampled(ncid, r, "albedo_chM7");
	n = nc_close(ncid);
	if(n != NC_NOERR)
		ncfatal(n, "nc_close failed for %s", path);

	medianBlur(sst, medf, 5);
	stdfilter(sst-medf, stdf, 7);
	nanblur(sst, blurf, 7);
	easyclouds = (sst < SST_LOW) | (stdf > STD_THRESH)
		| (abs(sst - blurf) > EDGE_THRESH);
	
	logprintf("gradmag...\n");
	gradientmag(sst, gradmag);

	logprintf("localmax...\n");
	localmax(gradmag, lam2, lam1, 1);

	easyfronts = (sst > SST_LOW) & (gradmag > 0.5)
		& (stdf < STD_THRESH) & (lam2 < -0.05);

	logprintf("delta...\n");
	delta = m15 - m16;
	omega = m14 - m15;

	logprintf("quantize sst delta...\n");
	quantize(lat, sst, delta, omega, gradmag, albedo, acspo, TQ, DQ, OQ, lut);
}

int
main(int argc, char **argv)
{
	Mat _glut, _lut, _cloud, _ocean;
	char *glut, *lut;
	int i, k, *cloud, *ocean;

	if(argc < 2)
		eprintf("usage: %s granule ...\n", argv[0]);
	
	// compute count of cloud/ocean entries in LUT
	k = 1;
	do{
		logprintf("granule %d: %s\n", k, argv[k]);
		granulelut(argv[k], _glut);
		CV_Assert(_glut.type() == CV_8SC1);
		
		if(_cloud.data == NULL){
			_cloud.create(_glut.dims, _glut.size, CV_32SC1);
			_cloud = Scalar(0);
			_ocean.create(_glut.dims, _glut.size, CV_32SC1);
			_ocean = Scalar(0);
		}
		glut = (char*)_glut.data;
		cloud = (int*)_cloud.data;
		ocean = (int*)_ocean.data;
		for(i = 0; i < (int)_glut.total(); i++){
			if(glut[i] == LUT_CLOUD)
				cloud[i]++;
			if(glut[i] == LUT_OCEAN)
				ocean[i]++;
		}
	}while(++k < argc);
	
	// compute final LUT based on majority rule
	_lut.create(_cloud.dims, _cloud.size, CV_8SC1);
	_lut = Scalar(LUT_UNKNOWN);
	lut = (char*)_lut.data;
	for(i = 0; i < (int)_cloud.total(); i++){
		lut[i] = LUT_UNKNOWN;
		if(cloud[i] > ocean[i])
			lut[i] = LUT_CLOUD;
		if(ocean[i] > cloud[i])
			lut[i] = LUT_OCEAN;
	}
	SAVENPY(_lut);
}

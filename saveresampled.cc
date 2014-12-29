//
// Program that resamples variables in a granule
//

#include "spt.h"

void
writevar(int ncid, const char *name, Mat &mat)
{
	int n, vid;
	
	n = nc_inq_varid(ncid, name, &vid);
	if(n != NC_NOERR)
		ncfatal(n, "nc_inq_varid failed for variable %s", name);

	switch(mat.type()){
	default:
		eprintf("invalid type %s", type2str(mat.type()));
	case CV_32FC1:
		n = nc_put_var_float(ncid, vid, (float*)mat.data);
		break;
	case CV_8UC1:
		n = nc_put_var_uchar(ncid, vid, (uchar*)mat.data);
		break;
	}
	if (n != NC_NOERR)
		ncfatal(n, "nc_put_var_float failed");
}

int
main(int argc, char **argv)
{
	Mat lat, slon, acspo, sst, sind;
	int ncid, n;
	char *path;
	Resample r;

	if(argc != 2)
		eprintf("usage: %s granule\n", argv[0]);
	path = argv[1];
	
	n = nc_open(path, NC_WRITE, &ncid);
	if (n != NC_NOERR)
		ncfatal(n, "nc_open failed for %s", path);
	
	readvar(ncid, "latitude", lat);
	readvar(ncid, "longitude", slon);
	readvar(ncid, "acspo_mask", acspo);
	readvar(ncid, "sst_regression", sst);
	
	resample_init(&r, lat, acspo);
	resample_float32(&r, sst, sst, true);
	slon = resample_sort(r.sind, slon);
	
	writevar(ncid, "acspo_mask", r.sacspo);
	writevar(ncid, "latitude", r.slat);
	writevar(ncid, "longitude", slon);
	writevar(ncid, "sst_regression", sst);
	
	n = nc_close(ncid);
	if(n != NC_NOERR)
		ncfatal(n, "nc_close failed for %s", path);
	
	return 0;
}

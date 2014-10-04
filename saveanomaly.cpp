#include "spt.h"

void
saveanomaly(char *path, Mat &mat)
{
	int n, ncid, dims[2], vid;
	
	CV_Assert(mat.channels() == 1);
	if(mat.depth() != CV_32F)
		mat.convertTo(mat, CV_32F);
	
	n = nc_create(path, NC_NOCLOBBER|NC_NETCDF4, &ncid);
	if (n != NC_NOERR)
		ncfatal(n, "nc_create failed for %s", path);
	n = nc_def_dim(ncid, "scan_lines_along_track", mat.rows, &dims[0]);
	if (n != NC_NOERR)
		ncfatal(n, "nc_def_dim failed");
	n = nc_def_dim(ncid, "pixels_across_track", mat.cols, &dims[1]);
	if (n != NC_NOERR)
		ncfatal(n, "nc_def_dim failed");
	n = nc_def_var (ncid, "anomaly", NC_FLOAT, nelem(dims), dims, &vid);
	if (n != NC_NOERR)
		ncfatal(n, "nc_def_var failed");
	n = nc_put_var_float(ncid, vid, (float*)mat.data);
	if (n != NC_NOERR)
		ncfatal(n, "nc_put_var_float failed");
	n = nc_close(ncid);
	if(n != NC_NOERR)
		ncfatal(n, "nc_close failed for %s", path);
}

int
main(int argc, char **argv)
{
	Mat lat, acspo, sst, reynolds, anomaly, rgb;
	int ncid, n;
	char *inpath, *outpath;

	if(argc != 3)
		eprintf("usage: %s granule outfile\n", argv[0]);
	inpath = argv[1];
	outpath = argv[2];
	
	n = nc_open(inpath, NC_NOWRITE, &ncid);
	if(n != NC_NOERR)
		ncfatal(n, "nc_open failed for %s", inpath);
	lat = readvar(ncid, "latitude");
	acspo = readvar(ncid, "acspo_mask");
	sst = readvar(ncid, "sst_regression");
	reynolds = readvar(ncid, "sst_reynolds");
	n = nc_close(ncid);
	if(n != NC_NOERR)
		ncfatal(n, "nc_close failed for %s", inpath);
	
	sst = resample_float64(sst, lat, acspo);
	anomaly = sst - reynolds;
	
	saveanomaly(outpath, anomaly);
	return 0;
}

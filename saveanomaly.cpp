#include "spt.h"

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
		ncfatal(n);
	lat = readvar(ncid, "latitude");
	acspo = readvar(ncid, "acspo_mask");
	sst = readvar(ncid, "sst_regression");
	reynolds = readvar(ncid, "sst_reynolds");
	n = nc_close(ncid);
	if(n != NC_NOERR)
		ncfatal(n);
	
	sst = resample_float64(sst, lat, acspo);
	anomaly = sst - reynolds;
	
	// TODO: write anomaly into a new netcdf file
	// instead of the following.
	resize(anomaly, anomaly, Size(), 0.20, 0.20);
	gray2rgb(anomaly, rgb, COLORMAP_JET);
	imwrite(outpath, rgb);
	
	//dumpmat("anomaly.bin", anomaly);

	return 0;
}

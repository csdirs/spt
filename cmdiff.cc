//
// Creates a cloud mask diff image comparing cloud mask in acspo_mask
// and spt_mask. The spt_mask variable in the NetCDF file is created by
// the spt program in the current directory.
//

#include "spt.h"

int
main(int argc, char *argv[])
{
	Mat acspo, spt, diff;
	int n, ncid;
	char *granule, *diffname;
	
	if(argc != 3)
		eprintf("usage: %s granule.nc diff.png\n", argv[0]);
	granule = argv[1];
	diffname = argv[2];
	n = nc_open(granule, NC_NOWRITE, &ncid);
	if(n != NC_NOERR)
		ncfatal(n, "nc_open failed for %s", granule);

	readvar(ncid, "acspo_mask", acspo);
	readvar(ncid, "spt_mask", spt);
	diffcloudmask(acspo, spt, diff);
	cvtColor(diff, diff, CV_RGB2BGR);
	resize(diff, diff, Size(), 1/6.0, 1/6.0, INTER_AREA);
	imwrite(diffname, diff);

	n = nc_close(ncid);
	if(n != NC_NOERR)
		ncfatal(n, "nc_close failed for %s", granule);
}

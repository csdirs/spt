#include "spt.h"

int
main(int argc, char **argv)
{
	Mat sst, reynolds, lat, lon, m14, m15, m16, elem, sstdil, sstero, rfilt, sstlap, medf, stdf, blurf;
	Mat m14gm, m15gm, m16gm;
	Mat acspo, gradmag, delta, omega, albedo, TQ, DQ, OQ, lut, glabels, feat, lam1, lam2, easyclouds, easyfronts;
	int ncid, n;
	char *path;
	Resample *r;

	if(argc != 2)
		eprintf("usage: %s granule\n", argv[0]);
	path = argv[1];
	logprintf("granule: %s\n", path);
	
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

SAVENPY(lat);
SAVENPY(lon);
SAVENPY(acspo);
SAVENPY(sst);
SAVENPY(albedo);

	medianBlur(sst, medf, 5);
	stdfilter(sst-medf, stdf, 7);
	nanblur(sst, blurf, 7);
	easyclouds = (sst < SST_LOW) | (stdf > STD_THRESH)
		| (abs(sst - blurf) > EDGE_THRESH);
SAVENPY(easyclouds);
	
	logprintf("gradmag...\n");
	gradientmag(sst, gradmag);
SAVENPY(gradmag);

	logprintf("localmax...\n");
	localmax(gradmag, lam2, lam1, 1);
SAVENPY(lam2);

	easyfronts = (sst > SST_LOW) & (gradmag > 0.5)
		& (stdf < STD_THRESH) & (lam2 < -0.05);
SAVENPY(easyfronts);

	logprintf("delta...\n");
	delta = m15 - m16;
	omega = m14 - m15;
SAVENPY(delta);
SAVENPY(omega);

	logprintf("quantize sst delta...\n");
	quantize(lat, sst, delta, omega, gradmag, albedo, acspo, TQ, DQ, OQ, lut);
}

//
// Resample and write back SST in a GHRSST granule.
//

#include "spt.h"

enum {
	L2PCloudMask = (1<<14)|(1<<15),
	L2PLandMask = 0x2,
};

int
read3dvar(int ncid, const char *name, Mat &dst)
{
	int varid;
	Mat tmp;
	
	varid = readvar(ncid, name, tmp);
	logprintf("tmp type: %s\n", type2str(tmp.type()));

	if(tmp.dims != 3 || tmp.size[0] != 1)
		abort();
		
	int esize = 0;
	switch(tmp.type()){
	default:
		abort();
	case CV_16SC1:
		esize = 2;
		break;
	case CV_32FC1:
		esize = 4;
		break;
	}
	dst.create(tmp.size[1], tmp.size[2], tmp.type());
	logprintf("dst rows %d cols %d\n", dst.rows, dst.cols);
	memmove(dst.data, tmp.data, esize*tmp.size[1]*tmp.size[2]);
	return varid;
}

void
roundmat(Mat &_src, Mat &_dst)
{
	CHECKMAT(_src, CV_32FC1);
	CHECKMAT(_dst, CV_32FC1);
	
	float *src = (float*)_src.data;
	float *dst = (float*)_dst.data;
	
	for(int i = 0; i < (int)_src.total(); i++)
		dst[i] = cvRound(src[i]);
}

// ./resam /tmp/20150325174000-STAR-L2P_GHRSST-SSTskin-VIIRS_NPP-ACSPO_V2.40-v02.0-fv01.0.nc
int
main(int argc, char **argv)
{
	Mat l2p_flags, lat, oldsst, newsst;
	int ncid, sstid, n;
	char *path;
	Resample r;

	if(argc != 2)
		eprintf("usage: %s granule\n", argv[0]);
	path = argv[1];
	logprintf("granule: %s\n", path);
	
	logprintf("reading and resampling...\n");
	n = nc_open(path, NC_WRITE, &ncid);
	if(n != NC_NOERR)
		ncfatal(n, "nc_open failed for %s", path);
	readvar(ncid, "lat", lat);
	read3dvar(ncid, "l2p_flags", l2p_flags);
	sstid = read3dvar(ncid, "sea_surface_temperature", oldsst);
	logprintf("l2p rows %d cols %d\n", l2p_flags.rows, l2p_flags.cols);
	
	// create fake acspo mask with only the land flag
	Mat acspo(l2p_flags.rows, l2p_flags.cols, CV_8UC1);
	uchar *dst = acspo.data;
	short *src = (short*)l2p_flags.data;
	for(int i = 0; i < (int)l2p_flags.total(); i++)
		dst[i] = (src[i] & L2PLandMask) << 1;

	float scale, offset;
	short fillvalue;
	n = nc_get_att_float(ncid, sstid, "scale_factor", &scale);
	if(n != NC_NOERR)
		ncfatal(n, "nc_get_att_float failed");
	n = nc_get_att_float(ncid, sstid, "add_offset", &offset);
	if(n != NC_NOERR)
		ncfatal(n, "nc_get_att_float failed");
	n = nc_get_att_short(ncid, sstid, "_FillValue", &fillvalue);
	if(n != NC_NOERR)
		ncfatal(n, "nc_get_att_schar failed");
	logprintf("scale %f, offset %f fillvalue %d\n", scale, offset, fillvalue);
	oldsst.convertTo(oldsst, CV_32FC1);
	oldsst.setTo(NAN, oldsst==fillvalue);
	oldsst = oldsst*scale + offset;

SAVENC(lat);
SAVENC(l2p_flags);
SAVENC(acspo);
SAVENC(oldsst);
	resample_init(&r, lat, acspo);
	resample_float32(&r, oldsst, newsst, true);

	newsst = (newsst - offset)/scale;
	roundmat(newsst, newsst);
	// set land to fill value
	newsst.setTo(fillvalue, (l2p_flags&L2PLandMask) == L2PLandMask);
	newsst.convertTo(newsst, CV_16SC1);
SAVENC(newsst);

	CHECKMAT(newsst, CV_16SC1);
	n = nc_put_var(ncid, sstid, newsst.data);
	if(n != NC_NOERR)
		ncfatal(n, "savenc: writing variable failed");
	n = nc_close(ncid);
	if(n != NC_NOERR)
		ncfatal(n, "savenc: closing %s failed", path);
}

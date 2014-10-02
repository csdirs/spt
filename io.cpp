#include "spt.h"

Mat
readvar(int ncid, const char *name)
{
	int i, varid, n, dimids[2];
	size_t shape[2];
	nc_type nct;
	Mat img;
	
	n = nc_inq_varid(ncid, name, &varid);
	if(n != NC_NOERR)
		ncfatal(n);
	n = nc_inq_var(ncid, varid, NULL, &nct, NULL, dimids, NULL);
	if(n != NC_NOERR)
		ncfatal(n);
	for(i = 0; i < 2; i++){
		n = nc_inq_dimlen(ncid, dimids[i], &shape[i]);
		if(n != NC_NOERR)
			ncfatal(n);
	}
	switch(nct){
	default:
		eprintf("unknown netcdf data type");
		break;
	case NC_BYTE:
		img = Mat::zeros(shape[0], shape[1], CV_8SC1);
		n = nc_get_var_schar(ncid, varid, (signed char*)img.data);
		if(n != NC_NOERR)
			ncfatal(n);
		break;
	case NC_UBYTE:
		img = Mat::zeros(shape[0], shape[1], CV_8UC1);
		n = nc_get_var_uchar(ncid, varid, (unsigned char*)img.data);
		if(n != NC_NOERR)
			ncfatal(n);
		break;
	case NC_FLOAT:
		img = Mat::zeros(shape[0], shape[1], CV_32FC1);
		n = nc_get_var_float(ncid, varid, (float*)img.data);
		if(n != NC_NOERR)
			ncfatal(n);
		img.convertTo(img, CV_64F);
		break;
	}
	return img;
}

void
dumpmat(const char *filename, Mat &m)
{
	int n;
	FILE *f;

	if(!m.isContinuous()){
		eprintf("m not continuous");
	}
	f = fopen(filename, "w");
	if(!f){
		eprintf("open failed");
	}
	n = fwrite(m.data, m.elemSize1(), m.rows*m.cols, f);
	if(n != m.rows*m.cols){
		fclose(f);
		printf("wrote %d items\n", n);
		eprintf("write failed");
	}
	fclose(f);
}

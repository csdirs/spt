#include "spt.h"

char*
fileprefix(const char *path)
{
	const char *b;
	char *p, *s;
	
	b = strrchr(path, '/');
	if(b == NULL)
		b = path;
	else
		b++;
	
	p = strdup(b);
	s = strrchr(p, '.');
	if(s != NULL)
		*s = '\0';
	return p;
}

Mat
readvar(int ncid, const char *name)
{
	int i, varid, n, dimids[2];
	size_t shape[2];
	nc_type nct;
	Mat img;
	
	n = nc_inq_varid(ncid, name, &varid);
	if(n != NC_NOERR)
		ncfatal(n, "nc_inq_varid failed for variable %s", name);
	n = nc_inq_var(ncid, varid, NULL, &nct, NULL, dimids, NULL);
	if(n != NC_NOERR)
		ncfatal(n, "nc_inq_var failed for variable %s", name);
	for(i = 0; i < 2; i++){
		n = nc_inq_dimlen(ncid, dimids[i], &shape[i]);
		if(n != NC_NOERR)
			ncfatal(n, "nc_inq_dimlen failed for dim %d", dimids[i]);
	}
	switch(nct){
	default:
		eprintf("unknown netcdf data type");
		break;
	case NC_BYTE:
		img = Mat::zeros(shape[0], shape[1], CV_8SC1);
		n = nc_get_var_schar(ncid, varid, (signed char*)img.data);
		if(n != NC_NOERR)
			ncfatal(n, "nc_get_var_schar failed");
		break;
	case NC_UBYTE:
		img = Mat::zeros(shape[0], shape[1], CV_8UC1);
		n = nc_get_var_uchar(ncid, varid, (unsigned char*)img.data);
		if(n != NC_NOERR)
			ncfatal(n, "nc_get_var_uchar failed");
		break;
	case NC_FLOAT:
		img = Mat::zeros(shape[0], shape[1], CV_32FC1);
		n = nc_get_var_float(ncid, varid, (float*)img.data);
		if(n != NC_NOERR)
			ncfatal(n, "nc_get_var_float failed");
		break;
	}
	return img;
}

void
savebin(const char *filename, Mat &m)
{
	int n;
	FILE *f;

	if(!m.isContinuous()){
		eprintf("m not continuous");
	}
	f = fopen(filename, "w");
	if(!f){
		eprintf("open %s failed:", filename);
	}
	n = fwrite(m.data, m.elemSize1(), m.rows*m.cols, f);
	if(n != m.rows*m.cols){
		fclose(f);
		eprintf("wrote %d/%d items; write failed:", n, m.rows*m.cols);
	}
	fclose(f);
}

void
savenc(const char *path, Mat &mat)
{
	enum {
		MAXDIMS = 5,
	};
	int i, n, ncid, dimids[MAXDIMS], varid, xtype;
	char *name;
	const char *dimnames[MAXDIMS] = {
		"dim0",
		"dim1",
		"dim2",
		"dim3",
		"dim4",
	};
	
	if(mat.dims > MAXDIMS)
		eprintf("savenc: too many dimensions %d\n", mat.dims);
	
	n = nc_create(path, NC_NETCDF4, &ncid);
	if(n != NC_NOERR)
		ncfatal(n, "savenc: creating %s failed", path);
	
	for(i = 0; i < mat.dims; i++){
		n = nc_def_dim(ncid, dimnames[i], mat.size[i], &dimids[i]);
		if(n != NC_NOERR)
			ncfatal(n, "savenc: creating dim %d failed", i);
	}
	
	xtype = -1;
	switch(mat.type()){
	default:
		eprintf("savenc: unsupported type %s\n", type2str(mat.type()));
		break;
	case CV_8UC1:	xtype = NC_UBYTE; break;
	case CV_8SC1:	xtype = NC_BYTE; break;
	case CV_16UC1:	xtype = NC_USHORT; break;
	case CV_16SC1:	xtype = NC_SHORT; break;
	case CV_32SC1:	xtype = NC_INT; break;
	case CV_32FC1:	xtype = NC_FLOAT; break;
	case CV_64FC1:	xtype = NC_DOUBLE; break;
	}
	
	n = nc_def_var(ncid, "data", xtype, mat.dims, dimids, &varid);
	if(n != NC_NOERR)
		ncfatal(n, "savenc: creating variable failed");
	
	if(0){	// enable compression?
		n = nc_def_var_deflate(ncid, varid, 0, 1, 1);
		if(n != NC_NOERR)
			ncfatal(n, "savenc: setting deflate parameters failed");
	}

	n = nc_put_var(ncid, varid, mat.data);
	if(n != NC_NOERR)
		ncfatal(n, "savenc: writing variable failed");
	
	n = nc_close(ncid);
	if(n != NC_NOERR)
		ncfatal(n, "savenc: closing %s failed", path);

	name = fileprefix(path);
	printf("%s = loadnc(\"%s\")\n", name, path);
	free(name);
}

// Print out error for NetCDF error number n and exit the program.
void
ncfatal(int n, const char *fmt, ...)
{
	va_list args;

	fflush(stdout);
	va_start(args, fmt);
	vfprintf(stderr, fmt, args);
	va_end(args);

	fprintf(stderr, ": %s\n", nc_strerror(n));
	exit(2);
}

// Open NetCDF file at path and initialize r.
int
open_resampled(const char *path, Resample *r)
{
	int ncid, n;
	Mat lat, acspo;
	
	n = nc_open(path, NC_NOWRITE, &ncid);
	if(n != NC_NOERR)
		ncfatal(n, "nc_open failed for %s", path);
	acspo = readvar(ncid, "acspo_mask");
	lat = readvar(ncid, "latitude");
	
	resample_init(r, lat, acspo);
	return ncid;
}

// Read a variable named name from NetCDF dataset ncid,
// resample the image if necessary and return it.
Mat
readvar_resampled(int ncid, Resample *r, const char *name)
{
	Mat img;
	
	if(strcmp(name, "latitude") == 0)
		return r->slat;
	if(strcmp(name, "acspo_mask") == 0)
		return r->sacspo;

	img = readvar(ncid, name);
	if(strcmp(name, "longitude") == 0){
		resample_sort(r->sind, img);
		return img;
	}
	
	logprintf("resampling %s...\n", name);
	resample_float32(r, img, img);
	return img;
}

#include "spt.h"

enum {
	MAXDIMS = 5,
};

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

void
readvar(int ncid, const char *name, Mat &img)
{
	int i, varid, n, ndims, dimids[MAXDIMS], ishape[MAXDIMS], cvt;
	size_t shape[MAXDIMS];
	nc_type nct;
	
	n = nc_inq_varid(ncid, name, &varid);
	if(n != NC_NOERR)
		ncfatal(n, "nc_inq_varid failed for variable %s", name);

	n = nc_inq_var(ncid, varid, NULL, &nct, &ndims, dimids, NULL);
	if(n != NC_NOERR)
		ncfatal(n, "nc_inq_var failed for variable %s", name);
	if(ndims > MAXDIMS)
		eprintf("number of dimensions %d > MAXDIMS=%d\n", ndims, MAXDIMS);
	
	for(i = 0; i < ndims; i++){
		n = nc_inq_dimlen(ncid, dimids[i], &shape[i]);
		if(n != NC_NOERR)
			ncfatal(n, "nc_inq_dimlen failed for dim %d", dimids[i]);
	}
	
	cvt = -1;
	switch(nct){
	default:
		eprintf("unknown netcdf data type");
		break;
	case NC_BYTE:	cvt = CV_8SC1; break;
	case NC_UBYTE:	cvt = CV_8UC1; break;
	case NC_SHORT:	cvt = CV_16SC1; break;
	case NC_USHORT:	cvt = CV_16UC1; break;
	case NC_INT:	cvt = CV_32SC1; break;
	case NC_FLOAT:	cvt = CV_32FC1; break;
	case NC_DOUBLE:	cvt = CV_64FC1; break;
	}
	
	for(i = 0; i < ndims; i++)
		ishape[i] = (int)shape[i];
	
	img.create(ndims, ishape, cvt);
	n = nc_get_var(ncid, varid, img.data);
	if(n != NC_NOERR)
		ncfatal(n, "readvar: nc_get_var failed");
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

void
loadnc(const char *path, Mat &mat)
{
	int n, ncid;
	
	n = nc_open(path, NC_NOWRITE, &ncid);
	if(n != NC_NOERR)
		ncfatal(n, "loadnc: opening %s failed", path);
	
	readvar(n, "data", mat);
	nc_close(n);
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
	readvar(ncid, "acspo_mask", acspo);
	readvar(ncid, "latitude", lat);
	
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

	readvar(ncid, name, img);
	if(strcmp(name, "longitude") == 0){
		resample_sort(r->sind, img);
		return img;
	}
	
	logprintf("resampling %s...\n", name);
	resample_float32(r, img, img);
	return img;
}

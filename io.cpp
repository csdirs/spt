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

static void
put2(uint16_t v, uchar *a)
{
	a[0] = v & 0xFF;
	a[1] = (v >> 8) & 0xFF;
}

static const char*
npytype(Mat &mat)
{
	switch(mat.type()){
	default:	return NULL;
	case CV_8UC1:	return "u1"; break;
	case CV_8SC1:	return "i1"; break;
	case CV_16UC1:	return "u2"; break;
	case CV_16SC1:	return "i2"; break;
	case CV_32SC1:	return "i4"; break;
	case CV_32FC1:	return "f4"; break;
	case CV_64FC1:	return "f8"; break;
	}
}	

static int
bigendian()
{
	uint32_t n = 0x04030201;
	return ((uchar*)&n)[0] == 4;
}

static uchar NPY_MAGIC[] = {0x93, 'N', 'U', 'M', 'P', 'Y', 0x01, 0x00};

void
savenpy(const char *filename, Mat &mat)
{
	FILE *f;
	int n, npad, nprefix;
	char hdr[200], pad[16], *name;
	uchar len[2];
	const char *type;
	
	type = npytype(mat);
	if(type == NULL)
		eprintf("unsupported type: %s\n", type2str(mat.type()));
	
	if(mat.dims > 3)
		eprintf("too many dimensions\n");
	
	if(mat.dims == 3){
		snprintf(hdr, nelem(hdr),
			"{'descr': '%c%s', 'fortran_order': False, 'shape': (%d, %d, %d),}",
			bigendian() ? '>' : '<', type, mat.size[0], mat.size[1], mat.size[2]);
	}else{
		snprintf(hdr, nelem(hdr),
			"{'descr': '%c%s', 'fortran_order': False, 'shape': (%d, %d),}",
			bigendian() ? '>' : '<', type, mat.rows, mat.cols);
	}
	hdr[nelem(hdr)-1] = '\0';
	
	// magic + header length + header + '\n'
	nprefix = nelem(NPY_MAGIC) + 2 + strlen(hdr) + 1;
	
	// create the padding required for the header so that
	// the matrix data is 16-byte aligned
	npad = ((nprefix+16-1)/16)*16 - nprefix;
	memset(pad, ' ', npad);
	pad[npad] = '\0';

	// length of header + pad + '\n'
	put2(strlen(hdr) + npad + 1, len);
	
	
	f = fopen(filename, "w");
	if(f == NULL)
		eprintf("fopen %s:", filename);
	
	n = fwrite(NPY_MAGIC, sizeof(*NPY_MAGIC), nelem(NPY_MAGIC), f);
	n += fwrite(len, sizeof(*len), nelem(len), f);
	n += fprintf(f, "%s%s\n", hdr, pad);
	if(n != nprefix+npad){
		fclose(f);
		eprintf("wrote failed:");
	}
	n = fwrite(mat.data, mat.elemSize1(), mat.total(), f);
	if(n != (int)mat.total()){
		fclose(f);
		eprintf("wrote %d/%d items; write failed:", n, mat.total());
	}
	fclose(f);

	name = fileprefix(filename);
	printf("%s = np.load(\"%s\")\n", name, filename);
	free(name);
}

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

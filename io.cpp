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

static uint16_t
get2(uchar *a)
{
	return (uint16_t(a[1])<<8) | a[0];
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

// Read Numpy .npy file from filename and save it in mat.
void
savenpy(const char *filename, Mat &mat)
{
	FILE *f;
	int i, n, npad, nprefix;
	char hdr[200], pad[16], *name, *p;
	uchar len[2];
	const char *type;
	
	type = npytype(mat);
	if(type == NULL)
		eprintf("unsupported type: %s\n", type2str(mat.type()));
	
	if(mat.dims == 0)
		eprintf("zero dimensions\n");
	
	// TODO: use snprintf to protect against buffer overflow
	p = hdr;
	p += sprintf(hdr, "{'descr': '%c%s', 'fortran_order': False, 'shape': (%d",
			bigendian() ? '>' : '<', type, mat.size[0]);
	if(mat.dims > 1){
		for(i = 1; i < mat.dims; i++)
			p += sprintf(p, ", %d", mat.size[i]);
	}
	p += sprintf(p, "),}");
	*p = '\0';
	
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

enum {
	TRUE = 256,
	FALSE,
	STRING,
	INT,
};

typedef struct Token Token;
struct  Token {
	int id;
	union {
		char *name;
		int n;
	} v;
};

static char*
get_token(char *s, Token *t)
{
	char *p;
	
	while(isspace(*s))
		s++;
	if(*s == '\0')
		return NULL;

	switch(*s){
	case '{':
	case '}':
	case '(':
	case ')':
	case ':':
	case ',':
		t->id = *s++;
		break;
	case '\'':
		p = ++s;
		s = strchr(s, '\'');
		if(*s != '\'')
			eprintf("unterminated queted string\n");
		*s++ = '\0';
		t->id = STRING;
		t->v.name = p;
		break;
	default:
		if(strncmp(s, "True", 4) == 0){
			t->id = TRUE;
			s += 4;
			break;
		}
		if(strncmp(s, "False", 5) == 0){
			t->id = FALSE;
			s += 5;
			break;
		}
		if('0' <= *s && *s <= '9'){
			t->id = INT;
			t->v.n = strtol(s, &s, 10);
			break;
		}
		eprintf("invalid token starting with %c (%d)\n", *s, *s);
	}
	return s;
}

static char*
eat_token(char *s, Token *t, int id)
{
	s = get_token(s, t);
	if(t->id != id)
		eprintf("unexpected token %d; want %d\n", t->id, id);
	return s;
}

typedef struct NPYHeader NPYHeader;
struct NPYHeader {
	int descr;
	int fortran_order;
	int shape[4];
	int nshape;
};

static int
cvtype(char *s){
	if(strlen(s) != 3)
		eprintf("invalid dtype %d\n", s);
	
	if(s[0] != '<')
		eprintf("big-endian dtype %s not supported\n", s);
	
	switch(s[1]){
	case 'i':
		switch(s[2]){
		case '1':	return CV_8SC1;
		case '2':	return CV_16SC1;
		case '4':	return CV_32SC1;
		}
		break;
	case 'u':
		switch(s[2]){
		case '1':	return CV_8UC1;
		case '2':	return CV_16UC1;
		}
		break;
	case 'f':
		switch(s[2]){
		case '4':	return CV_32FC1;
		case '8':	return CV_64FC1;
		}
		break;
	}
	eprintf("unsupported dtype %s\n", s);
	abort();	/* unreachable */
}

char*
parse_descr(char *s, NPYHeader *hdr)
{
	Token t;
	
	s = eat_token(s, &t, STRING);
	hdr->descr = cvtype(t.v.name);
	return s;
}

static char*
parse_tuple(char *s, NPYHeader *hdr)
{
	int i;
	Token t;
	
	s = eat_token(s, &t, '(');
	
	hdr->nshape = 0;
	for(i = 0; i < (int)nelem(hdr->shape); i++){
		s = get_token(s, &t);
		if(t.id == ')')
			break;
		if(t.id != INT)
			eprintf("expected INT token; got %d\n", t.id);
		hdr->shape[hdr->nshape++] = t.v.n;

		s = get_token(s, &t);
		if(t.id == ')')
			break;
		if(t.id != ',')
			eprintf("expected ','; got %d\n", t.id);
	}
	if(t.id != ')')
		eprintf("too many numbers in tuple\n");
	return s;
}

static char*
parse_dict(char *s, NPYHeader *hdr)
{
	Token t;

	s = eat_token(s, &t, '{');
	for(;;){
		s = get_token(s, &t);
		if(t.id != STRING)
			break;
	
		if(strcmp(t.v.name, "descr") == 0){
			s = eat_token(s, &t, ':');
			s = parse_descr(s, hdr);
		} else if(strcmp(t.v.name, "fortran_order") == 0){
			s = eat_token(s, &t, ':');
			s = get_token(s, &t);
			if(t.id != TRUE && t.id != FALSE)
				eprintf("expected boolean; got %d\n", t.id);
			hdr->fortran_order = t.id == TRUE;
		} else if(strcmp(t.v.name, "shape") == 0){
			s = eat_token(s, &t, ':');
			s = parse_tuple(s, hdr);
		} else {
			eprintf("unexpected key %s\n", t.v.name);
		}
		s = eat_token(s, &t, ',');
	}
	if(t.id != '}')
		eprintf("expected '}'; got %d\n", t.id);
	return s;
}

void
loadnpy(const char *filename, Mat &mat)
{
	FILE *f;
	uchar buf[200];
	int n, len;
	NPYHeader hdr;
	
	f = fopen(filename, "r");
	if(f == NULL)
		eprintf("fopen %s:", filename);
	
	// read and check NPY magic at the beginning of the file
	n = fread(buf, sizeof(uchar), nelem(NPY_MAGIC), f);
	if(n < (int)nelem(NPY_MAGIC))
		eprintf("short read:");
	if(memcmp(buf, NPY_MAGIC, nelem(NPY_MAGIC)) != 0)
		eprintf("%s: not an NPY file\n", filename);
	
	// read header length
	n = fread(buf, sizeof(uchar), 2, f);
	if(n < 2)
		eprintf("short read:");
	len = get2(buf);
	
	// read header string
	n = fread(buf, sizeof(uchar), len, f);
	if(n < len)
		eprintf("short read:");
	buf[len] = '\0';
	
	// parse header
	parse_dict((char*)buf, &hdr);
	//printf("descr = %d %d\n", hdr.descr, CV_8SC1);
	//printf("descr = %d\n", hdr.fortran_order);
	//for(int i = 0; i < hdr.nshape; i++)
	//	printf("shape[%d] = %d\n", i, hdr.shape[i]);
	
	// read Mat data
	mat.create(hdr.nshape, hdr.shape, hdr.descr);
	n = fread(mat.data, mat.elemSize(), mat.total(), f);
	if(n < (int)(mat.elemSize()*mat.total()))
		eprintf("short read:");
	fclose(f);
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

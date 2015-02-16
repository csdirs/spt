//
// Utility functions
//

#include "spt.h"

void
eprintf(const char *fmt, ...)
{
	va_list args;

	fflush(stdout);
	va_start(args, fmt);
	vfprintf(stderr, fmt, args);
	va_end(args);

	if(fmt[0] != '\0' && fmt[strlen(fmt)-1] == ':')
		fprintf(stderr, " %s", strerror(errno));
	fprintf(stderr, "\n");

	exit(2);
}

void
logprintf(const char *fmt, ...)
{
	va_list args;
	time_t now;
	char *t;

	time(&now);
	t = ctime(&now);
	// omit '\n' from time when printing
	printf("# %.*s ", (int)strlen(t)-1, t);
	
	fflush(stdout);
	va_start(args, fmt);
	vprintf(fmt, args);
	va_end(args);
	fflush(stdout);
}

char*
estrdup(const char *s)
{
	char *dup;

	dup = strdup(s);
	if(dup == NULL)
		eprintf("strdup of \"%s\" failed:", s);
	return dup;
}

void*
emalloc(size_t n)
{
	void *buf;

	buf = malloc(n);
	if(buf == NULL)
		eprintf("malloc failed:");
	return buf;
}

const char*
type2str(int type)
{
	switch(type){
	default:       return "UnknownType"; break;
	case CV_8UC1:  return "CV_8UC1"; break;
	case CV_8SC1:  return "CV_8SC1"; break;
	case CV_16UC1: return "CV_16UC1"; break;
	case CV_16SC1: return "CV_16SC1"; break;
	case CV_32SC1: return "CV_32SC1"; break;
	case CV_32FC1: return "CV_32FC1"; break;
	case CV_64FC1: return "CV_64FC1"; break;
	}
}

// Cloud mask values
enum {
	CMClear,
	CMProbably,
	CMSure,
	CMInvalid,
};

// Number of bits in cloud mask
enum {
	CMBits = 2,
};

enum {
	White	= 0xFFFFFF,
	Red		= 0xFF0000,
	Green	= 0x00FF00,
	Blue	= 0x0000FF,
	Yellow	= 0xFFFF00,
	JetRed	= 0x7F0000,
	JetBlue	= 0x00007F,
	JetGreen	= 0x7CFF79,
};

#define SetColor(v, c) do{ \
		(v)[0] = ((c)>>16) & 0xFF; \
		(v)[1] = ((c)>>8) & 0xFF; \
		(v)[2] = ((c)>>0) & 0xFF; \
	}while(0);

// Compute RGB diff image of cloud mask.
//
// _old -- old cloud mask (usually ACSPO cloud mask)
// _new -- new cloud mask (usually SPT cloud mask)
// _rgb -- RGB diff image (output)
//
void
diffcloudmask(const Mat &_old, const Mat &_new, Mat &_rgb)
{
	int i;
	uchar *old, *new1, *rgb, oval, nval;
	
	CHECKMAT(_old, CV_8UC1);
	CHECKMAT(_new, CV_8UC1);
	
	_rgb.create(_old.size(), CV_8UC3);
	rgb = _rgb.data;
	old = _old.data;
	new1 = _new.data;
	
	for(i = 0; i < (int)_old.total(); i++){
		oval = old[i]>>MaskCloudOffset;
		nval = new1[i] & 0x03;
		
		if(oval == CMProbably)
			oval = CMSure;
		if(nval == CMProbably)
			nval = CMSure;
		
		switch((oval<<CMBits) | nval){
		default:
			SetColor(rgb, Yellow);
			break;
		
		case (CMInvalid<<CMBits) | CMInvalid:
			SetColor(rgb, White);
			break;
		
		case (CMClear<<CMBits) | CMClear:
			SetColor(rgb, JetBlue);
			break;
		
		case (CMSure<<CMBits) | CMSure:
			SetColor(rgb, JetRed);
			break;
		
		case (CMSure<<CMBits) | CMClear:
		case (CMInvalid<<CMBits) | CMClear:
			SetColor(rgb, JetGreen);
			break;
		}
		rgb += 3;
	}
}

// Return a filename based on granule path path with suffix suf.
// e.g. savefilename("/foo/bar/qux.nc", ".png") returns "qux.png"
//
char*
savefilename(char *path, const char *suf)
{
	int n;
	char buf[200], *p;
	
	p = strrchr(path, '/');
	if(!p)
		p = path;
	else
		p++;
	
	n = strlen(p) - 3;
	p = strncpy(buf, p, n);	// don't copy ".nc" extension
	p += n;
	strcpy(p, suf);
	return estrdup(buf);
}

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

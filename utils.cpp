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
	printf("%.*s ", (int)strlen(t)-1, t);
	
	fflush(stdout);
	va_start(args, fmt);
	vfprintf(stderr, fmt, args);
	va_end(args);
	printf("\n");

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

void
gray2rgb(Mat &src, Mat &dst, int cmap)
{
	double min, max;

	minMaxLoc(src, &min, &max, NULL, NULL);

	dst = src.clone();
	switch(src.type()){
	default:
		eprintf("unkown Mat type %s", type2str(src.type()));
		break;
	case CV_8SC1:
	case CV_8UC1:
	case CV_16SC1:
	case CV_16UC1:
	case CV_32SC1:
		dst.convertTo(dst, CV_64F);
		// fallthrough
	case CV_32FC1:
	case CV_64FC1:
		dst -= min;
		dst.convertTo(dst, CV_8U, 255.0/(max-min));
		break;
	}
	//resize(dst, tmp2, Size(), scale, scale);
	applyColorMap(dst, dst, cmap);
}


void
cmapimshow(string name, Mat &img, int cmap)
{
	Mat rgb;
	
	gray2rgb(img, rgb, cmap);

	//namedWindow(name, WINDOW_AUTOSIZE);
	namedWindow(name, CV_WINDOW_NORMAL|CV_WINDOW_KEEPRATIO);
	imshow(name, rgb);
}

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

	switch(src.type()){
	default:
		eprintf("unkown Mat type %s", type2str(src.type()));
		break;
	case CV_8SC1:
	case CV_8UC1:
	case CV_16SC1:
	case CV_16UC1:
	case CV_32SC1:
		src.convertTo(src, CV_64F);
		// fallthrough
	case CV_32FC1:
	case CV_64FC1:
		src -= min;
		src.convertTo(src, CV_8U, 255.0/(max-min));
		break;
	}
	//resize(src, tmp2, Size(), scale, scale);
	applyColorMap(src, dst, cmap);
}


void
cmapimshow(string name, Mat &img, int cmap)
{
	gray2rgb(img, img, cmap);

	//namedWindow(name, WINDOW_AUTOSIZE);
	namedWindow(name, CV_WINDOW_NORMAL|CV_WINDOW_KEEPRATIO);
	imshow(name, img);
}

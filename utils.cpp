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

char*
type2str(int type)
{
	int n;
	const char *dp;
	char cn, r[10];
	
	switch(type & CV_MAT_DEPTH_MASK){
	default: dp = "User"; break;
	case CV_8U:  dp = "8U"; break;
	case CV_8S:  dp = "8S"; break;
	case CV_16U: dp = "16U"; break;
	case CV_16S: dp = "16S"; break;
	case CV_32S: dp = "32S"; break;
	case CV_32F: dp = "32F"; break;
	case CV_64F: dp = "64F"; break;
	}
	cn = (1 + (type >> CV_CN_SHIFT)) + '0';
	
	n = snprintf(r, sizeof(r)-1, "%sC%c", dp, cn);
	r[n] = '\0';
	
	return estrdup(r);
}

void
cmapimshow(string name, Mat &img, int cmap)
{
	double min, max;

	minMaxLoc(img, &min, &max, NULL, NULL);

	switch(img.type()){
	default:
		eprintf("unkown Mat type %s", type2str(img.type()));
		break;
	case CV_8SC1:
	case CV_8UC1:
	case CV_16SC1:
	case CV_16UC1:
	case CV_32SC1:
		img.convertTo(img, CV_64F);
		// fallthrough
	case CV_32FC1:
	case CV_64FC1:
		img -= min;
		img.convertTo(img, CV_8U, 255.0/(max-min));
		break;
	}
	//resize(img, tmp2, Size(), scale, scale);
	applyColorMap(img, img, cmap);

	//namedWindow(name, WINDOW_AUTOSIZE);
	namedWindow(name, CV_WINDOW_NORMAL|CV_WINDOW_KEEPRATIO);
	imshow(name, img);
}

#include "spt.h"

void
ncfatal(int n)
{
	cerr << "Error: " << nc_strerror(n) << endl;
	exit(2);
}

void
fatal(string msg)
{
	cerr << msg << endl;
	exit(2);
}

void
cmapImshow(string name, Mat &img, int cmap)
{
	double min, max;

	minMaxLoc(img, &min, &max, NULL, NULL);
	cout << name << " min: " << min << " max: " << max << endl;
	cout << name << " type: " << img.type() << endl;

	switch(img.type()){
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

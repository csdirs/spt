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

string
type2str(int type) {
  string r;

  switch(type & CV_MAT_DEPTH_MASK){
    default:     r = "User"; break;
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
  }
  r += "C";
  r += (1 + (type >> CV_CN_SHIFT)) + '0';
  return r;
}

void
checktype(Mat &mat, string name, int type){
	if(mat.type() != type){
		cerr << "type of Mat " << name
			<< " expected " << type2str(type)
			<< ", got " << type2str(mat.type()) << endl;
		exit(2);
	}
}

void
cmapimshow(string name, Mat &img, int cmap)
{
	double min, max;

	minMaxLoc(img, &min, &max, NULL, NULL);
	cout << name << " min: " << min << " max: " << max << endl;
	cout << name << " type: " << img.type() << endl;

	switch(img.type()){
	default:
		fatal("unkown Mat type");
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

void
dumpmat(const char *filename, Mat &m)
{
	int n;
	FILE *f;

	if(!m.isContinuous()){
		fatal("m not continuous");
	}
	f = fopen(filename, "w");
	if(!f){
		fatal("open failed");
	}
	printf("rows=%d, cols=%d, elemSize=%ld\n", m.rows, m.cols, m.elemSize());
	n = fwrite(m.data, m.elemSize1(), m.rows*m.cols, f);
	if(n != m.rows*m.cols){
		fclose(f);
		printf("wrote %d items\n", n);
		fatal("write failed");
	}
	fclose(f);
}


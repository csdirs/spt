#include "spt.h"

#define GRAD_THRESH 0.3
#define GRAD_LOW 0.1
#define SST_LOW 270
#define EDGE_THRESH 1

#define TQ_STEP 2
#define DQ_STEP 0.5

void
clipsst(Mat &sst)
{
	float *p;
	int i;

	p = (float*)sst.data;
	for(i = 0; i < (int)sst.total(); i++){
		if(p[i] > 1000 || p[i] < -1000)
			p[i] = NAN;
	}
}

void
laplacian(Mat &src, Mat &dst)
{
	 Mat kern = (Mat_<double>(3,3) <<
	 	0,     1/4.0,  0,
		1/4.0, -4/4.0, 1/4.0,
		0,     1/4.0,  0);
	filter2D(src, dst, -1, kern);
}

void
avgfilter(Mat &src, Mat &dst, int ksize)
{
	Mat kern = Mat::ones(ksize, ksize, CV_64FC1);
	kern *= 1.0/(ksize*ksize);
	filter2D(src, dst, -1, kern);
}

void
gradientmag(Mat &img, Mat &gm)
{
    Mat h = (Mat_<double>(5,1) <<
		0.036420, 0.248972, 0.429217, 0.248972, 0.036420);
    Mat hp = (Mat_<double>(5,1) <<
		0.108415, 0.280353, 0, -0.280353, -0.108415);
	Mat tmp, dX, dY, ht, hpt;

	// TODO: padding needed here?
	transpose(h, ht);
	transpose(hp, hpt);
	filter2D(img, tmp, -1, hp);
	filter2D(tmp, dX, -1, ht);
	filter2D(img, tmp, -1, h);
	filter2D(tmp, dY, -1, hpt);
	sqrt(dX.mul(dX) + dY.mul(dY), gm);
}

enum {
	NStd = 3,
};

void
localmax(Mat &gradmag, Mat &high, Mat &low, int sigma)
{
	int i, j, x, y, winsz;
	double e, a, dd, mu1, mu2, *Dxxp, *Dxyp, *Dyyp, *highp, *lowp;
	Mat DGaussxx, DGaussxy, DGaussyy,
		Dxx, Dxy, Dyy;
	
	CV_Assert(gradmag.type() == CV_64FC1);

	winsz = 2*(NStd*sigma) + 1;
	DGaussxx = Mat::zeros(winsz, winsz, CV_64FC1);
	DGaussxy = Mat::zeros(winsz, winsz, CV_64FC1);
	DGaussyy = Mat::zeros(winsz, winsz, CV_64FC1);

	for(i = 0; i < winsz; i++){
		x = i - NStd*sigma;
		for(j = 0; j < winsz; j++){
			y = j - NStd*sigma;

			e = exp(-(SQ(x) + SQ(y)) / (2*SQ(sigma)));
			DGaussxx.at<double>(i, j) =
				DGaussyy.at<double>(j, i) =
				1/(2*M_PI*pow(sigma, 4)) *
				(SQ(x)/SQ(sigma) - 1) * e;
			DGaussxy.at<double>(i, j) =
				1/(2*M_PI*pow(sigma, 6)) * (x*y) * e;
		}
	}
	filter2D(gradmag, Dxx, -1, DGaussxx);
	filter2D(gradmag, Dxy, -1, DGaussxy);
	filter2D(gradmag, Dyy, -1, DGaussyy);

	high.create(Dxx.rows, Dxx.cols, CV_64FC1);
	low.create(Dxx.rows, Dxx.cols, CV_64FC1);
	highp = (double*)high.data;
	lowp = (double*)low.data;
	Dxxp = (double*)Dxx.data;
	Dxyp = (double*)Dxy.data;
	Dyyp = (double*)Dyy.data;
	for(i = 0; i < Dxx.rows*Dxx.cols; i++){
		a = Dxxp[i] + Dyyp[i];
		dd = sqrt(SQ(Dxxp[i] - Dyyp[i]) + 4*SQ(Dxyp[i]));
		mu1 = 0.5*(a + dd);
		mu2 = 0.5*(a - dd);
		if(abs(mu1) > abs(mu2)){
			highp[i] = mu1;
			lowp[i] = mu2;
		}else{
			highp[i] = mu2;
			lowp[i] = mu1;
		}
	}
}

char*
savefilename(char *path)
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
	strcpy(p, ".png");
	return estrdup(buf);
}

void
findfronts(Mat &sst, Mat &gradmag)
{
	Mat avgsst, lam2, lam1, D, easyclouds, easyfronts, maskf, labels, stats, centoids;
	int i, nlabels;
	
	logprintf("avgfilter...\n");
	avgfilter(sst, avgsst, 7);
dumpmat("avgsst.bin", avgsst);

	logprintf("localmax...\n");
	localmax(gradmag, lam2, lam1, 1);
dumpmat("lam2.bin", lam2);

	D = sst - avgsst;
dumpmat("D.bin", D);
	easyclouds = (sst < 270) | (gradmag > GRAD_THRESH) | (abs(D) > EDGE_THRESH);
dumpmat("easyclouds.bin", easyclouds);

	easyfronts = (sst > 270) & (gradmag > GRAD_THRESH) & (abs(D) < EDGE_THRESH)
		& (lam2 < -0.05);
dumpmat("easyfronts.bin", easyfronts);

	maskf = (easyclouds != 0) & (easyfronts == 0);
dumpmat("maskf.bin", maskf);

	logprintf("connected components...\n");
	nlabels = connectedComponentsWithStats(maskf, labels, stats, centoids, 8, CV_32S);
dumpmat("labels.bin", labels);
	logprintf("number of connected components: %d\n", nlabels);
	for(i = 0; i < min(10, nlabels); i++){
		logprintf("connected component %d area: %d\n", i, stats.at<int>(i, CC_STAT_AREA));
	}
}

void
quantize_sst_delta(const Mat &_sst, const Mat &_gradmag, const Mat &_delta, Mat &TQ, Mat &DQ)
{
	int i;
	double *sst, *gm, *delta;
	short *tq, *dq;
	
	CV_Assert(_sst.type() == CV_64FC1);
	CV_Assert(_gradmag.type() == CV_64FC1);
	CV_Assert(_delta.type() == CV_64FC1);
	
	TQ.create(_sst.size(), CV_16SC1);
	DQ.create(_sst.size(), CV_16SC1);
	
	sst = (double*)_sst.data;
	gm = (double*)_gradmag.data;
	delta = (double*)_delta.data;
	tq = (short*)TQ.data;
	dq = (short*)DQ.data;
	for(i = 0; i < (int)_sst.total(); i++){
		tq[i] = dq[i] = -1;
		
		if((gm[i] < GRAD_LOW) & (sst[i] > SST_LOW) & (delta[i] > -0.5)){
			tq[i] = cvRound((sst[i] - SST_LOW) / TQ_STEP);
			dq[i] = cvRound((delta[i] + 1) / DQ_STEP);
		}
	}
}

void
quantized_features(Mat &TQ, Mat &DQ, Mat &_lat, Mat &_lon, Mat &_sst, Mat &_delta)
{
	int i, t, d, tqmax, dqmax, nlabels;
	Mat _mask, labels, stats, centoids;
	short *tq, *dq;
	uchar *mask;
	
	CV_Assert(TQ.type() == CV_16SC1);
	CV_Assert(DQ.type() == CV_16SC1);
	CV_Assert(_lat.type() == CV_64FC1);
	CV_Assert(_lon.type() == CV_64FC1);
	CV_Assert(_sst.type() == CV_64FC1);
	CV_Assert(_delta.type() == CV_64FC1);
	
	// compute max of tq (tqmax) and max of dq (dqmax)
	tq = (short*)TQ.data;
	dq = (short*)DQ.data;
	tqmax = tq[0];
	dqmax = dq[0];
	for(i = 1; i < (int)TQ.total(); i++){
		if(tq[i] > tqmax)
			tqmax = tq[i];
		if(dq[i] > dqmax)
			dqmax = dq[i];
	}
	
	_mask.create(TQ.size(), CV_8UC1);
	mask = (uchar*)_mask.data;
	for(t = 0; t < tqmax; t++){
		for(d = 0; d < dqmax; d++){
			// create mask for (t, d)
			for(i = 0; i < (int)_mask.total(); i++)
				mask[i] = tq[i] == t && dq[i] == d ? 255 : 0;
			
			nlabels = connectedComponentsWithStats(_mask, labels, stats, centoids, 8, CV_32S);
			//dumpmat
		}
	}
}

int
main(int argc, char **argv)
{
	Mat sst, lat, m15, m16, elem, sstdil, sstero, rfilt, sstlap, sind;
	Mat acspo, landmask, gradmag, delta, TQ, DQ;
	int ncid, n;
	char *path;

	if(argc != 2)
		eprintf("usage: %s granule\n", argv[0]);
	path = argv[1];
	
	n = nc_open(path, NC_NOWRITE, &ncid);
	if(n != NC_NOERR)
		ncfatal(n, "nc_open failed for %s", path);
	sst = readvar(ncid, "sst_regression");
	lat = readvar(ncid, "latitude");
	acspo = readvar(ncid, "acspo_mask");
	m15 = readvar(ncid, "brightness_temp_chM15");
	m16 = readvar(ncid, "brightness_temp_chM16");
	n = nc_close(ncid);
	if(n != NC_NOERR)
		ncfatal(n, "nc_close failed for %s", path);

	logprintf("resampling...\n");
	sst = resample_float32(sst, lat, acspo);
	sst.convertTo(sst, CV_64F);
savenpy("sst.npy", sst);
return 0;

	m15 = resample_float32(m15, lat, acspo);
	m16 = resample_float32(m16, lat, acspo);
	delta = m15 - m16;
	delta.convertTo(delta, CV_64F);
dumpmat("m15.bin", m15);
dumpmat("m16.bin", m16);
dumpmat("delta.bin", delta);
	
	logprintf("gradmag...\n");
	gradientmag(sst, gradmag);
dumpmat("gradmag.bin", gradmag);

	quantize_sst_delta(sst, gradmag, delta, TQ, DQ);
dumpmat("TQ.bin", TQ);
dumpmat("DQ.bin", DQ);



/*
	logprintf("dilate...");
	elem = getStructuringElement(MORPH_RECT, Size(3, 3), Point(1, 1));
	dilate(interpsst, sstdil, elem);
	logprintf("erode...");
	erode(interpsst, sstero, elem);
	logprintf("rangefilt...");
	subtract(sstdil, sstero, rfilt);

	logprintf("laplacian...");
	laplacian(interpsst, sstlap);

	clipsst(rfilt);
	cmapimshow("Rangefilt SST", rfilt, COLORMAP_JET);
	cmapimshow("Laplacian SST", sstlap, COLORMAP_JET);
	cmapimshow("acspo", acspo, COLORMAP_JET);
	cmapimshow("interpsst", interpsst, COLORMAP_JET);
	cmapimshow("gradmag", gradmag, COLORMAP_JET);
	cmapimshow("high", high, COLORMAP_JET);
	cmapimshow("low", low, COLORMAP_JET);

	namedWindow("SortIdx SST", CV_WINDOW_NORMAL|CV_WINDOW_KEEPRATIO);
	imshow("SortIdx SST", sind);
*/
/*
	cmapimshow("SST", interpsst, COLORMAP_JET);
	cmapimshow("blur(SSt)", blursst, COLORMAP_JET);

	while(waitKey(0) != 'q')
		;
*/

	return 0;
}

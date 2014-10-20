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
savenpy("avgsst.npy", avgsst);

	logprintf("localmax...\n");
	localmax(gradmag, lam2, lam1, 1);
savenpy("lam2.npy", lam2);

	D = sst - avgsst;
savenpy("D.npy", D);
	easyclouds = (sst < 270) | (gradmag > GRAD_THRESH) | (abs(D) > EDGE_THRESH);
savenpy("easyclouds.npy", easyclouds);

	easyfronts = (sst > 270) & (gradmag > GRAD_THRESH) & (abs(D) < EDGE_THRESH)
		& (lam2 < -0.05);
savenpy("easyfronts.npy", easyfronts);

	maskf = (easyclouds != 0) & (easyfronts == 0);
savenpy("maskf.npy", maskf);

	logprintf("connected components...\n");
	nlabels = connectedComponentsWithStats(maskf, labels, stats, centoids, 8, CV_32S);
savenpy("labels.npy", labels);
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
	Mat _mask, _labels, stats, centoids, _bigcomp, _feat;
	int *labels;
	double *lat, *lon, *feat_lat, *feat_lon;
	short *tq, *dq;
	uchar *mask, *bigcomp;
	char name[100];
	
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
	_bigcomp.create(_sst.total(), 1, CV_8UC1);
	_feat.create(5, _sst.total(), CV_64FC1);
	
logprintf("lat rows=%d cols=%d total=%d; sst rows=%d cols=%d total=%d\n",
_lat.rows, _lat.cols, _lat.total(), _sst.rows, _sst.cols, _sst.total());
	
	mask = (uchar*)_mask.data;
	bigcomp = (uchar*)_bigcomp.data;
	lat = (double*)_lat.data;
	lon = (double*)_lon.data;
	feat_lat = (double*)_feat.ptr(0);
	feat_lon = (double*)_feat.ptr(1);
	
	for(t = 0; t < tqmax; t++){
		for(d = 0; d < dqmax; d++){
			// create mask for (t, d)
			for(i = 0; i < (int)_mask.total(); i++)
				mask[i] = tq[i] == t && dq[i] == d ? 255 : 0;
			
			nlabels = connectedComponentsWithStats(_mask, _labels, stats, centoids, 8, CV_32S);
			if(nlabels <= 1)
				continue;
			labels = (int*)_labels.data;
			//snprintf(name, nelem(name), "labels_t%02d_d%02d.npy", t, d);
			//savenpy(name, labels);
			printf("# t=%2d/%d, d=%2d/%d, nlabels=%d\n",
				t+1, tqmax, d+1, dqmax, nlabels);
		
			for(i = 0; i < nlabels; i++)
				bigcomp[i] = stats.at<int>(i, CC_STAT_AREA) >= 200 ? 255: 0;
			
			for(i = 0; i < (int)_sst.total(); i++){
				if(!mask[i] || !bigcomp[labels[i]])
					continue;
				lon[i] = lat[i];
				feat_lat[i] = lat[i];
				feat_lon[i] = lon[i];
			}
		}
		fflush(stdout);
	}
}

int
main(int argc, char **argv)
{
	Mat sst, lat, lon, m15, m16, elem, sstdil, sstero, rfilt, sstlap, sind;
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
	lon = readvar(ncid, "longitude");
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

	m15 = resample_float32(m15, lat, acspo);
	m16 = resample_float32(m16, lat, acspo);
	delta = m15 - m16;
	delta.convertTo(delta, CV_64F);
	lat.convertTo(lat, CV_64F);
	lon.convertTo(lon, CV_64F);
savenpy("m15.npy", m15);
savenpy("m16.npy", m16);
savenpy("delta.npy", delta);
	
	logprintf("gradmag...\n");
	gradientmag(sst, gradmag);
savenpy("gradmag.npy", gradmag);

	quantize_sst_delta(sst, gradmag, delta, TQ, DQ);
savenpy("TQ.npy", TQ);
savenpy("DQ.npy", DQ);
	quantized_features(TQ, DQ, lat, lon, sst, delta);



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

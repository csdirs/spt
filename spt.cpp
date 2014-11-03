#include "spt.h"

#define GRAD_THRESH 0.3
#define GRAD_LOW 0.1
#define SST_LOW 270
#define EDGE_THRESH 1

#define TQ_STEP 3
#define DQ_STEP 0.5

enum {
	FEAT_LAT,
	FEAT_LON,
	FEAT_SST,
	FEAT_DELTA,
	NFEAT,
};

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
	double e, a, dd, mu1, mu2;
	float *Dxxp, *Dxyp, *Dyyp, *highp, *lowp;
	Mat DGaussxx, DGaussxy, DGaussyy,
		Dxx, Dxy, Dyy;
	
	CV_Assert(gradmag.type() == CV_32FC1);

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

	CV_Assert(Dxx.type() == CV_32FC1);
	CV_Assert(Dxy.type() == CV_32FC1);
	CV_Assert(Dyy.type() == CV_32FC1);
	
	high.create(Dxx.rows, Dxx.cols, CV_32FC1);
	low.create(Dxx.rows, Dxx.cols, CV_32FC1);
	highp = (float*)high.data;
	lowp = (float*)low.data;
	Dxxp = (float*)Dxx.data;
	Dxyp = (float*)Dxy.data;
	Dyyp = (float*)Dyy.data;
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
	float *sst, *gm, *delta;
	short *tq, *dq;
	
	CV_Assert(_sst.type() == CV_32FC1);
	CV_Assert(_gradmag.type() == CV_32FC1);
	CV_Assert(_delta.type() == CV_32FC1);
	
	TQ.create(_sst.size(), CV_16SC1);
	DQ.create(_sst.size(), CV_16SC1);
	
	sst = (float*)_sst.data;
	gm = (float*)_gradmag.data;
	delta = (float*)_delta.data;
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

// Run connected component for t == tq and d == dq, and save the features
// for the connected components in _feat.
//	size -- size of image
//	t -- quantized SST value
//	d -- quantized delta value
//	tq, dq -- quantized SST, delta
//	sst, delta, lat, lon -- original SST, delta, latitude, longitude
//	glab -- global label assigned to _labels for (t == tq && d == dq)
//	glabels -- global labels (output)
//	_feat -- features (output)
int
quantized_features_td(Size size, int t, int d, const short *tq, const short *dq,
	const float *sst, const float *delta, const float *lat, const float *lon, int glab,
	int *glabels, Mat &_feat)
{
	Mat _mask, _cclabels, stats, centoids, _bigcomp, _count, _avgsst, _avgdelta;
	double *avgsst, *avgdelta;
	float *feat_lat, *feat_lon, *feat_sst, *feat_delta;
	int i, ncc, lab, *cclabels, *count;
	uchar *mask, *bigcomp;
	
	// create mask for (t, d)
	_mask.create(size, CV_8UC1);
	mask = (uchar*)_mask.data;
	for(i = 0; i < (int)_mask.total(); i++)
		mask[i] = tq[i] == t && dq[i] == d ? 255 : 0;
	
	ncc = connectedComponentsWithStats(_mask, _cclabels, stats, centoids, 8, CV_32S);
	if(ncc <= 1)
		return 0;
//printf("# t=%2d, d=%2d, ncc=%d\n", t+1, d+1, ncc);

	_bigcomp.create(ncc, 1, CV_8UC1);
	_count.create(ncc, 1, CV_32SC1);
	_avgsst.create(ncc, 1, CV_64FC1);
	_avgdelta.create(ncc, 1, CV_64FC1);

	cclabels = (int*)_cclabels.data;
	bigcomp = (uchar*)_bigcomp.data;
	count = (int*)_count.data;
	avgsst = (double*)_avgsst.data;
	avgdelta = (double*)_avgdelta.data;
	
	for(lab = 0; lab < ncc; lab++)
		bigcomp[lab] = stats.at<int>(lab, CC_STAT_AREA) >= 200 ? 255: 0;
	
	memset(count, 0, sizeof(*count)*ncc);
	memset(avgsst, 0, sizeof(*avgsst)*ncc);
	memset(avgdelta, 0, sizeof(*avgdelta)*ncc);
	
	for(i = 0; i < size.area(); i++){
		lab = cclabels[i];
		if(mask[i] && bigcomp[lab]
		&& !isnan(sst[i]) && !isnan(delta[i])){
			avgsst[lab] += sst[i];
			avgdelta[lab] += delta[i];
			count[lab]++;
		}
	}
	for(lab = 0; lab < ncc; lab++){
		if(bigcomp[lab]){
			avgsst[lab] /= count[lab];
			avgdelta[lab] /= count[lab];
		}
	}
	feat_lat = (float*)_feat.ptr(FEAT_LAT);
	feat_lon = (float*)_feat.ptr(FEAT_LON);
	feat_sst = (float*)_feat.ptr(FEAT_SST);
	feat_delta = (float*)_feat.ptr(FEAT_DELTA);
	
	for(i = 0; i < size.area(); i++){
		lab = cclabels[i];
		if(mask[i] && bigcomp[lab]){
			feat_lat[i] = lat[i];
			feat_lon[i] = lon[i];
			feat_sst[i] = avgsst[lab];
			feat_delta[i] = avgdelta[lab];
			glabels[i] = glab + lab;
		}
	}
	return ncc;
}

void
quantized_features(Mat &TQ, Mat &DQ, Mat &_lat, Mat &_lon, Mat &_sst, Mat &_delta,
	Mat &_glabels, Mat &_feat)
{
	int i, glab, tqmax, dqmax, *glabels;
	float *lat, *lon, *sst, *delta, *feat;
	short *tq, *dq;
	
	CV_Assert(TQ.type() == CV_16SC1);
	CV_Assert(DQ.type() == CV_16SC1);
	CV_Assert(_lat.type() == CV_32FC1);
	CV_Assert(_lon.type() == CV_32FC1);
	CV_Assert(_sst.type() == CV_32FC1);
	CV_Assert(_delta.type() == CV_32FC1);
	
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
	
	_glabels.create(_sst.size(), CV_32SC1);
	_feat.create(NFEAT, _sst.total(), CV_32FC1);
	
//logprintf("lat rows=%d cols=%d total=%d; sst rows=%d cols=%d total=%d\n",
//	_lat.rows, _lat.cols, _lat.total(), _sst.rows, _sst.cols, _sst.total());
	
	lat = (float*)_lat.data;
	lon = (float*)_lon.data;
	feat = (float*)_feat.data;
	sst = (float*)_sst.data;
	delta = (float*)_delta.data;
	glabels = (int*)_glabels.data;
	
	for(i = 0; i < (int)_feat.total(); i++)
		feat[i] = NAN;
	for(i = 0; i < (int)_glabels.total(); i++)
		glabels[i] = -1;
	
	glab = 0;
	for(int t = 0; t < tqmax; t++){
		for(int d = 0; d < dqmax; d++){
			glab += quantized_features_td(_sst.size(), t, d, tq, dq,
				sst, delta, lat, lon, glab, glabels, _feat);
		}
	}
	transpose(_feat, _feat);
}

static void
nnlabel(Mat &_feat, Mat &_lat, Mat &_lon, Mat &_sst, Mat &_delta, Mat &_sstclust)
{
	int i, k, *indices;
	float *vs, *vd, *lat, *lon, *sst, *delta, *sstclust;
	Mat _indices;
	std::vector<float> q(NFEAT), dists(1);
	std::vector<int> ind(1);
	flann::SearchParams sparam(4);
	
	CV_Assert(_feat.type() == CV_32FC1 && _feat.isContinuous()
		&& _feat.cols == NFEAT);
	CV_Assert(_lat.type() == CV_32FC1 && _lat.isContinuous());
	CV_Assert(_lon.type() == CV_32FC1 && _lon.isContinuous());
	CV_Assert(_sst.type() == CV_32FC1 && _sst.isContinuous());
	CV_Assert(_delta.type() == CV_32FC1 && _delta.isContinuous());
	
	// copy SST before we remove all the NaNs
	_sstclust = _feat.col(FEAT_SST).clone();
	sstclust = (float*)_sstclust.data;

	// remove NaNs from features
	_indices.create(_feat.rows, 1, CV_32SC1);
	indices = (int*)_indices.data;
	k = 0;
	for(i = 0; i < _feat.rows; i++){
		vs = (float*)_feat.ptr(i);
		if(!isnan(vs[0]) && i != k){
			vd = (float*)_feat.ptr(k);
			memmove(vd, vs, NFEAT*sizeof(*vd));
			indices[k] = i;
			k++;
		}
	}
	
	_feat = _feat.rowRange(0, k);
	logprintf("building nearest neighbor indices...\n");
	flann::Index idx(_feat, flann::KMeansIndexParams(16, 1));
	logprintf("searching nearest neighbor indices...\n");
	
	lat = (float*)_lat.data;
	lon = (float*)_lon.data;
	sst = (float*)_sst.data;
	delta = (float*)_delta.data;

	for(i = 0; i < (int)_sst.total(); i++){
		if(isnan(sstclust[i]) && !isnan(sst[i])){
			q[FEAT_LAT] = lat[i];
			q[FEAT_LON] = lon[i];
			q[FEAT_SST] = sst[i];
			q[FEAT_DELTA] = delta[i];
			idx.knnSearch(q, ind, dists, 1, sparam);
			sstclust[i] = sstclust[indices[ind[0]]];
		}
	}
}

int
main(int argc, char **argv)
{
	Mat sst, reynolds, lat, lon, m15, m16, anomaly, elem, sstdil, sstero, rfilt, sstlap, sind;
	Mat acspo, landmask, gradmag, delta, TQ, DQ, glabels, feat, sstclust, lam1, lam2;
	int ncid, n;
	char *path;
	Resample *r;

	if(argc != 2)
		eprintf("usage: %s granule\n", argv[0]);
	path = argv[1];
	
	n = nc_open(path, NC_NOWRITE, &ncid);
	if(n != NC_NOERR)
		ncfatal(n, "nc_open failed for %s", path);
	sst = readvar(ncid, "sst_regression");
	reynolds = readvar(ncid, "sst_reynolds");
	lat = readvar(ncid, "latitude");
	lon = readvar(ncid, "longitude");
	acspo = readvar(ncid, "acspo_mask");
	m15 = readvar(ncid, "brightness_temp_chM15");
	m16 = readvar(ncid, "brightness_temp_chM16");
	n = nc_close(ncid);
	if(n != NC_NOERR)
		ncfatal(n, "nc_close failed for %s", path);

	logprintf("resampling...\n");
savenpy("lat_before.npy", lat);
savenpy("acspo_before.npy", acspo);
	r = new Resample;
	resample_init(r, lat, acspo);
	resample_float32(r, sst, sst);
savenpy("lat_after.npy", lat);
savenpy("acspo_after.npy", acspo);
	resample_float32(r, m15, m15);
	resample_float32(r, m16, m16);
	delete r;

	logprintf("anomaly and delta...\n");
	anomaly = sst - reynolds;
savenpy("sst.npy", sst);
savenpy("anomaly.npy", anomaly);
	delta = m15 - m16;
savenpy("delta.npy", delta);
	
	logprintf("gradmag...\n");
	gradientmag(sst, gradmag);
savenpy("gradmag.npy", gradmag);

	logprintf("localmax...\n");
	localmax(gradmag, lam2, lam1, 1);
savenpy("lam2.npy", lam2);

	logprintf("quantize sst delta...\n");
	quantize_sst_delta(sst, gradmag, delta, TQ, DQ);
savenpy("TQ.npy", TQ);
savenpy("DQ.npy", DQ);
	logprintf("quantized featured...\n");
	quantized_features(TQ, DQ, lat, lon, sst, delta, glabels, feat);
savenpy("glabels.npy", glabels);
savenpy("feat.npy", feat);

	nnlabel(feat, lat, lon, sst, delta, sstclust);
savenpy("sstclust.npy", sstclust);

//easyfronts = (sst > 270) & (gradmag > 0.3) & (lam2 < -0.01)

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

	logprintf("done\n");
	return 0;
}

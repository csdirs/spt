#include "spt.h"

#define GRAD_THRESH 0.3
#define GRAD_LOW 0.1
#define SST_LOW 270
#define SST_HIGH 315
#define DELTA_LOW -3
#define DELTA_HIGH 3
#define EDGE_THRESH 1
#define STD_THRESH 0.5

#define TQ_STEP 3
#define DQ_STEP 0.5

#define TQ_HIST_STEP 1
#define DQ_HIST_STEP 0.25


#define SCALE_LAT(x)	((x) * 10)
#define SCALE_LON(x)	((x) * 10)
#define SCALE_SST(x)	(x)
#define SCALE_DELTA(x)	((x) * 6)

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
	// TODO: replace easyclouds with this:
	//std = stdfilt(sst - medianBlur(sst, 5), 7);
	//easyclouds = (sst < 270) | (sst > STD_THRESH) | (abs(D) > EDGE_THRESH);
savenpy("easyclouds.npy", easyclouds);

	easyfronts = (sst > 270) & (gradmag > GRAD_THRESH) & (abs(D) < EDGE_THRESH)
		& (lam2 < -0.05);
savenpy("easyfronts.npy", easyfronts);

	maskf = (easyclouds == 0) & (easyfronts != 0);
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
quantize_sst_delta(const Mat &_sst, const Mat &_gradmag, const Mat &_delta, Mat &TQ, Mat &DQ, Mat &_hist, Mat &_fronthist)
{
	int i, hrows, hcols, th, dh, *hist, *fronthist;
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
	
	hrows = cvRound((SST_HIGH - SST_LOW) * (1.0/TQ_HIST_STEP)) + 1;
	hcols = cvRound((DELTA_HIGH - DELTA_LOW) * (1.0/DQ_HIST_STEP)) + 1;
	_hist.create(hrows, hcols, CV_32SC1);
	_fronthist.create(_sst.size(), CV_32SC1);
	
	hist = (int*)_hist.data;
	fronthist = (int*)_fronthist.data;
	memset(hist, 0, hrows*hcols*sizeof(*hist));
	memset(fronthist, 0, _sst.rows*_sst.cols*sizeof(*fronthist));
	
	for(i = 0; i < (int)_sst.total(); i++){
		tq[i] = dq[i] = -1;
		
		if(gm[i] < GRAD_LOW		// && delta[i] > -0.5
		&& !isnan(sst[i]) && !isnan(delta[i])
		&& SST_LOW < sst[i] && sst[i] < SST_HIGH
		&& DELTA_LOW < delta[i] && delta[i] < DELTA_HIGH){
			tq[i] = cvRound((sst[i] - SST_LOW) / TQ_STEP);
			dq[i] = cvRound((delta[i] - DELTA_LOW) / DQ_STEP);
		
			th = cvRound((sst[i] - SST_LOW) / TQ_HIST_STEP);
			dh = cvRound((delta[i] - DELTA_LOW) / DQ_HIST_STEP);
			hist[th*hcols + dh]++;
		}
	}
	for(i = 0; i < (int)_sst.total(); i++){
		if(gm[i] >= GRAD_LOW
		&& !isnan(sst[i]) && !isnan(delta[i])
		&& SST_LOW < sst[i] && sst[i] < SST_HIGH
		&& DELTA_LOW < delta[i] && delta[i] < DELTA_HIGH){
			th = cvRound((sst[i] - SST_LOW) / TQ_HIST_STEP);
			dh = cvRound((delta[i] - DELTA_LOW) / DQ_HIST_STEP);
			fronthist[i] = hist[th*hcols + dh];
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
	const float *sst, const float *delta, const float *lat, const float *lon,
	Mat &_cclabels, Mat &_feat)
{
	Mat _mask, stats, centoids, _ccrename, _count, _avgsst, _avgdelta;
	double *avgsst, *avgdelta;
	float *feat_lat, *feat_lon, *feat_sst, *feat_delta;
	int i, ncc, lab, newlab, *cclabels, *ccrename, *count;
	uchar *mask;
	
	// create mask for (t, d)
	_mask.create(size, CV_8UC1);
	mask = (uchar*)_mask.data;
	for(i = 0; i < (int)_mask.total(); i++)
		mask[i] = tq[i] == t && dq[i] == d ? 255 : 0;
	
	ncc = connectedComponentsWithStats(_mask, _cclabels, stats, centoids, 8, CV_32S);
	if(ncc <= 1)
		return 0;
//printf("# t=%2d, d=%2d, ncc=%d\n", t+1, d+1, ncc);

	_ccrename.create(ncc, 1, CV_32SC1);
	cclabels = (int*)_cclabels.data;
	ccrename = (int*)_ccrename.data;
	
	// Remove small connected components and rename labels to be contiguous.
	// Also, set background label 0 (where mask is 0) to -1.
	newlab = 0;
	ccrename[0] = -1;
	for(lab = 1; lab < ncc; lab++){
		if(stats.at<int>(lab, CC_STAT_AREA) >= 200)
			ccrename[lab] = newlab++;
		else
			ccrename[lab] = -1;
	}
	ncc = newlab;
	for(i = 0; i < size.area(); i++)
		cclabels[i] = ccrename[cclabels[i]];
	
	// remove these since they are wrong after the labels renaming
	stats.release();
	centoids.release();
	
	_count.create(ncc, 1, CV_32SC1);
	_avgsst.create(ncc, 1, CV_64FC1);
	_avgdelta.create(ncc, 1, CV_64FC1);
	count = (int*)_count.data;
	avgsst = (double*)_avgsst.data;
	avgdelta = (double*)_avgdelta.data;
	memset(count, 0, sizeof(*count)*ncc);
	memset(avgsst, 0, sizeof(*avgsst)*ncc);
	memset(avgdelta, 0, sizeof(*avgdelta)*ncc);
	
	for(i = 0; i < size.area(); i++){
		lab = cclabels[i];
		if(lab >= 0 && !isnan(sst[i]) && !isnan(delta[i])){
			avgsst[lab] += sst[i];
			avgdelta[lab] += delta[i];
			count[lab]++;
		}
	}
	for(lab = 0; lab < ncc; lab++){
		avgsst[lab] /= count[lab];
		avgdelta[lab] /= count[lab];
	}
	feat_lat = (float*)_feat.ptr(FEAT_LAT);
	feat_lon = (float*)_feat.ptr(FEAT_LON);
	feat_sst = (float*)_feat.ptr(FEAT_SST);
	feat_delta = (float*)_feat.ptr(FEAT_DELTA);
	
	for(i = 0; i < size.area(); i++){
		lab = cclabels[i];
		if(lab >= 0){
			feat_lat[i] = SCALE_LAT(lat[i]);
			feat_lon[i] = SCALE_LON(lon[i]);
			feat_sst[i] = SCALE_SST(avgsst[lab]);
			feat_delta[i] = SCALE_DELTA(avgdelta[lab]);
		}
	}
	return ncc;
}

// Cluster and find features.
// TQ, DQ -- quantized SST and delta images
// _lat, _lon, _sst, _delta -- latitude, longitude, SST, delta images
// _glabels -- global labels (output)
// _feat -- features (output)
void
quantized_features(const Mat &TQ, const Mat &DQ, const Mat &_lat, const Mat &_lon,
	const Mat &_sst, const Mat &_delta, Mat &_glabels, Mat &_feat)
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
	#pragma omp parallel for
	for(int t = 0; t < tqmax; t++){
		#pragma omp parallel for
		for(int d = 0; d < dqmax; d++){
			Mat _cclabels;
			int ncc, lab, *cclabels;
			
			ncc = quantized_features_td(_sst.size(), t, d, tq, dq,
				sst, delta, lat, lon, _cclabels, _feat);
			CV_Assert(_cclabels.type() == CV_32SC1 && _cclabels.isContinuous());
			cclabels = (int*)_cclabels.data;
			
			#pragma omp critical
			if(ncc > 0){
				for(i = 0; i < (int)_cclabels.total(); i++){
					lab = cclabels[i];
					if(lab >= 0)
						glabels[i] = glab + lab;
				}
				glab += ncc;
			}
		}
	}
	transpose(_feat, _feat);
}

// Update labels by nearest label training.
// _feat -- features (rows containing NaNs are removed)
// _lat, _lon, _sst, _delta -- latitude, longitude, SST, delta images
// _glabels -- global labels (output)
static void
nnlabel(Mat &_feat, const Mat &_lat, const Mat &_lon, const Mat &_sst, const Mat &_delta, Mat &_acspo, Mat &_glabels)
{
	int i, k, *indices, *glabels;
	float *vs, *vd, *lat, *lon, *sst, *delta;
	Mat _indices;
	std::vector<float> q(NFEAT), dists(1);
	std::vector<int> ind(1);
	flann::SearchParams sparam(4);
	uchar *acspo;
	
	CV_Assert(_feat.type() == CV_32FC1 && _feat.isContinuous()
		&& _feat.cols == NFEAT);
	CV_Assert(_lat.type() == CV_32FC1 && _lat.isContinuous());
	CV_Assert(_lon.type() == CV_32FC1 && _lon.isContinuous());
	CV_Assert(_sst.type() == CV_32FC1 && _sst.isContinuous());
	CV_Assert(_delta.type() == CV_32FC1 && _delta.isContinuous());
	CV_Assert(_acspo.type() == CV_8UC1 && _acspo.isContinuous());
	CV_Assert(_glabels.type() == CV_32SC1 && _delta.isContinuous());

	// Remove features (rows in _feat) containing NaNs.
	// There are two cases: either all the features are NaN or
	// none of the features are NaN.
	_indices.create(_feat.rows, 1, CV_32SC1);
	indices = (int*)_indices.data;
	k = 0;
	for(i = 0; i < _feat.rows; i++){
		vs = (float*)_feat.ptr(i);
		if(!isnan(vs[FEAT_LAT]) && i != k){
			vd = (float*)_feat.ptr(k);
			memmove(vd, vs, NFEAT*sizeof(*vd));
			indices[k] = i;
			k++;
		}
	}
	_feat = _feat.rowRange(0, k);
logprintf("reduced number of features: %d\n", k);
	
	logprintf("building nearest neighbor indices...\n");
	flann::Index idx(_feat, flann::KMeansIndexParams(16, 1));
	logprintf("searching nearest neighbor indices...\n");
	
	lat = (float*)_lat.data;
	lon = (float*)_lon.data;
	sst = (float*)_sst.data;
	delta = (float*)_delta.data;
	acspo = (uchar*)_acspo.data;
	glabels = (int*)_glabels.data;

	for(i = 0; i < (int)_sst.total(); i++){
		if(glabels[i] < 0 			//&& (acspo[i]&MaskCloud) != MaskCloudClear
		&& !isnan(sst[i]) && !isnan(delta[i])){
			q[FEAT_LAT] = SCALE_LAT(lat[i]);
			q[FEAT_LON] = SCALE_LON(lon[i]);
			q[FEAT_SST] = SCALE_SST(sst[i]);
			q[FEAT_DELTA] = SCALE_DELTA(delta[i]);
			idx.knnSearch(q, ind, dists, 1, sparam);
			if(dists[0] < 30)
				glabels[i] = glabels[indices[ind[0]]];
		}
	}
}

int
open_resampled(const char *path, Resample *r)
{
	int ncid, n;
	Mat lat, acspo;
	
	n = nc_open(path, NC_NOWRITE, &ncid);
	if(n != NC_NOERR)
		ncfatal(n, "nc_open failed for %s", path);
	acspo = readvar(ncid, "acspo_mask");
	lat = readvar(ncid, "latitude");
	
	resample_init(r, lat, acspo);
	return ncid;
}

Mat
readvar_resampled(int ncid, Resample *r, const char *name)
{
	Mat img;
	
	if(strcmp(name, "latitude") == 0)
		return r->slat;
	if(strcmp(name, "acspo_mask") == 0)
		return r->sacspo;

	img = readvar(ncid, name);
	if(strcmp(name, "longitude") == 0){
		resample_sort(r->sind, img);
		return img;
	}
	
	logprintf("resampling %s...\n", name);
	resample_float32(r, img, img);
	return img;
}

// Separable blur implementation that can handle images containing NaN.
// OpenCV blur does not correctly handle such images.
void
nanblur(const Mat &src, Mat &dst, int ksize)
{
	Mat kernel = Mat::ones(ksize, 1, CV_32FC1)/ksize;
	sepFilter2D(src, dst, -1, kernel, kernel);
}

// Standard deviation filter.
void
stdfilter(const Mat &src, Mat &dst, int ksize)
{
	int i;
	Mat b, bs, _tmp;
	float *tmp;
	
	// dst = sqrt(blur(img^2) - blur(img)^2)
	nanblur(src.mul(src), bs, ksize);
	nanblur(src, b, ksize);

	// avoid sqrt of nagative number
	_tmp = bs - b.mul(b);
	CV_Assert(_tmp.type() == CV_32FC1 && _tmp.isContinuous());
	tmp = (float*)_tmp.data;
	for(i = 0; i < (int)_tmp.total(); i++){
		if(tmp[i] < 0)
			tmp[i] = 0;
	}
	sqrt(_tmp, dst);
}

int
main(int argc, char **argv)
{
	Mat sst, reynolds, lat, lon, m15, m16, elem, sstdil, sstero, rfilt, sstlap, medblur, stdf;
	Mat acspo, gradmag, delta, TQ, DQ, hist, fronthist, glabels, feat, lam1, lam2;
	int ncid, n;
	char *path;
	Resample *r;

	if(argc != 2)
		eprintf("usage: %s granule\n", argv[0]);
	path = argv[1];
	
	
	logprintf("reading and resampling...\n");
	r = new Resample;
	ncid = open_resampled(path, r);
	sst = readvar_resampled(ncid, r, "sst_regression");
	lat = readvar_resampled(ncid, r, "latitude");
	lon = readvar_resampled(ncid, r, "longitude");
	acspo = readvar_resampled(ncid, r, "acspo_mask");
	m15 = readvar_resampled(ncid, r, "brightness_temp_chM15");
	m16 = readvar_resampled(ncid, r, "brightness_temp_chM16");
	n = nc_close(ncid);
	if(n != NC_NOERR)
		ncfatal(n, "nc_close failed for %s", path);

savenpy("lat.npy", lat);
savenpy("lon.npy", lon);
savenpy("acspo.npy", acspo);
savenpy("sst.npy", sst);

	medianBlur(sst, medblur, 5);
savenpy("medblur.npy", medblur);

	stdfilter(sst-medblur, stdf, 7);
savenpy("stdf.npy", stdf);


	logprintf("delta...\n");
	delta = m15 - m16;
savenpy("delta.npy", delta);
	
	logprintf("gradmag...\n");
	gradientmag(sst, gradmag);
savenpy("gradmag.npy", gradmag);

	logprintf("localmax...\n");
	localmax(gradmag, lam2, lam1, 1);
savenpy("lam2.npy", lam2);

	logprintf("quantize sst delta...\n");
	quantize_sst_delta(sst, gradmag, delta, TQ, DQ, hist, fronthist);
savenpy("TQ.npy", TQ);
savenpy("DQ.npy", DQ);
savenpy("hist.npy", hist);
savenpy("fronthist.npy", fronthist);
	logprintf("quantized featured...\n");
	quantized_features(TQ, DQ, lat, lon, sst, delta, glabels, feat);
savenpy("glabels.npy", glabels);
savenpy("feat.npy", feat);

	nnlabel(feat, lat, lon, sst, delta, acspo, glabels);
savenpy("glabels_nn.npy", glabels);

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
*/
/*
	cmapimshow("SST", interpsst, COLORMAP_JET);
	cmapimshow("blur(SSt)", blursst, COLORMAP_JET);

	while(waitKey(0) != 'q')
		;
*/

	logprintf("done\n");
	delete r;
	return 0;
}

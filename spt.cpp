#include "spt.h"

#define GRAD_THRESH 0.3
#define GRAD_LOW 0.1
#define SST_LOW 270
#define SST_HIGH 309
#define DELTA_LOW -1
#define DELTA_HIGH 3
#define OMEGA_LOW -5	// TODO: compute from data
#define OMEGA_HIGH 0	// TODO: compute from data
#define ALBEDO_LOW 3
#define ALBEDO_HIGH 4
#define EDGE_THRESH 1
#define STD_THRESH 0.5

#define TQ_STEP 3
#define DQ_STEP 0.5
#define OQ_STEP 0.5

#define TQ_HIST_STEP 1
#define DQ_HIST_STEP 0.25
#define OQ_HIST_STEP OQ_STEP

#define SCALE_LAT(x)	((x) * 10)
#define SCALE_LON(x)	((x) * 10)
#define SCALE_SST(x)	(x)
#define SCALE_DELTA(x)	((x) * 6)

#define SAVENPY(X)	savenpy(#X ".npy", (X))

enum {
	FEAT_LAT,
	FEAT_LON,
	FEAT_SST,
	FEAT_DELTA,
	NFEAT,
};

enum {
	DEBUG = 1,
	
	LUT_UNKNOWN = -1,
	LUT_OCEAN = 0,
	LUT_CLOUD = 1,
	
	LUT_LAT_SPLIT = 4,
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

// Separable blur implementation that can handle images containing NaN.
// OpenCV blur does not correctly handle such images.
void
nanblur(const Mat &src, Mat &dst, int ksize)
{
	Mat kernel = Mat::ones(ksize, 1, CV_64FC1)/ksize;
	sepFilter2D(src, dst, -1, kernel, kernel);
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
	// TODO: use sepFilter2D instead of filter2D
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

// savefilename returns a filename based on granule path path
// with suffix sub.
// e.g. savefilename("/foo/bar/qux.nc", ".png") returns "qux.png"
char*
savefilename(char *path, const char *suf)
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
	strcpy(p, suf);
	return estrdup(buf);
}

// Quantize SST and delta values.
// _sst, _delta, _omega -- SST, delta, and omega images
// _gradmag, _albedo -- gradient magnitude and albedo image
// TQ, DQ, OQ -- quantized SST, delta, omega respectively (output)
// lut -- look up table for cloud/ocean in quantization space (output)
void
quantize(const Mat &_lat, const Mat &_sst, const Mat &_delta, Mat &_omega,
	const Mat &_gradmag, Mat &_albedo, Mat &_acspo,
	Mat &TQ, Mat &DQ, Mat &OQ, Mat &lut)
{
	int i, j, k, ncloud[LUT_LAT_SPLIT], nocean[LUT_LAT_SPLIT];
	float *lat, *sst, *delta, *omega, *gm, *albedo, la;
	double o, c;
	short *tq, *dq, *oq, li;
	uchar *acspo;
	Mat ocean, cloud;
	
	CV_Assert(_sst.type() == CV_32FC1);
	CV_Assert(_delta.type() == CV_32FC1);
	CV_Assert(_omega.type() == CV_32FC1);
	CV_Assert(_gradmag.type() == CV_32FC1);
	CV_Assert(_albedo.type() == CV_32FC1);
	CV_Assert(_acspo.type() == CV_8UC1);
	
	TQ.create(_sst.size(), CV_16SC1);
	DQ.create(_sst.size(), CV_16SC1);
	OQ.create(_sst.size(), CV_16SC1);
	
	lat = (float*)_lat.data;
	sst = (float*)_sst.data;
	delta = (float*)_delta.data;
	omega = (float*)_omega.data;
	gm = (float*)_gradmag.data;
	albedo = (float*)_albedo.data;
	acspo = _acspo.data;
	tq = (short*)TQ.data;
	dq = (short*)DQ.data;
	oq = (short*)OQ.data;
	
	// allocate space for LUT and initilize all entries to -1
	const int lutsizes[] = {
		LUT_LAT_SPLIT,
		cvRound((SST_HIGH - SST_LOW) * (1.0/TQ_STEP)) + 1,
		cvRound((DELTA_HIGH - DELTA_LOW) * (1.0/DQ_STEP)) + 1,
		cvRound((OMEGA_HIGH - OMEGA_LOW) * (1.0/OQ_STEP)) + 1,
	};
	cloud.create(4, lutsizes, CV_32SC1);
	cloud = Scalar(0);
	ocean.create(4, lutsizes, CV_32SC1);
	ocean = Scalar(0);
	lut.create(4, lutsizes, CV_8SC1);
	lut = Scalar(LUT_UNKNOWN);
	
	logprintf("LUT size is %dx%dx%d\n", lut.size[0], lut.size[1], lut.size[2]);
	
	// quantize SST and delta, and also computer the histogram
	// of counts per quantization bin
	for(li = 0; li < LUT_LAT_SPLIT; li++)
		ncloud[li] = nocean[li] = 0;
	for(i = 0; i < (int)_sst.total(); i++){
		tq[i] = dq[i] = oq[i] = -1;
		
		if(gm[i] < GRAD_LOW		// && delta[i] > -0.5
		&& !isnan(sst[i]) && !isnan(delta[i])
		&& SST_LOW < sst[i] && sst[i] < SST_HIGH
		&& DELTA_LOW < delta[i] && delta[i] < DELTA_HIGH
		&& OMEGA_LOW < omega[i] && omega[i] < OMEGA_HIGH){
			tq[i] = cvRound((sst[i] - SST_LOW) / TQ_STEP);
			dq[i] = cvRound((delta[i] - DELTA_LOW) / DQ_STEP);
			oq[i] = cvRound((omega[i] - OMEGA_LOW) / OQ_STEP);
			la = abs(lat[i]);
			if(la < 30){
				li = 0;
			}else if(la < 45){
				li = 1;
			}else if(la < 60){
				li = 2;
			}else{
				li = 3;
			}
			
			if((acspo[i] & MaskGlint) == 0){
				int idx[] = {li, tq[i], dq[i], oq[i]};
				if(albedo[i] > 8){
					cloud.at<int>(idx) += 1;
					ncloud[li]++;
				}
				if(albedo[i] < 3){
					ocean.at<int>(idx) += 1;
					nocean[li]++;
				}
			}
		}
	}
	
SAVENPY(ocean);
SAVENPY(cloud);
	for(li = 0; li < lutsizes[0]; li++){
		for(i = 0; i < lutsizes[1]; i++){
			for(j = 0; j < lutsizes[2]; j++){
				for(k = 0; k < lutsizes[3]; k++){
					int idx[] = {li, i, j, k};
					o = ocean.at<int>(idx) / (double)nocean[li];
					c = cloud.at<int>(idx) / (double)ncloud[li];
					if(o > c)
						lut.at<char>(idx) = LUT_OCEAN;
					if(c > o)
						lut.at<char>(idx) = LUT_CLOUD;
				}
			}
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
	
	// TODO:
	// - average omega and lat per cluster
	// - query LUT and disregard clusters that are cloud
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

// Remove features from _feat that are not on the border of clusters defined
// by the clustering labels in _glabels.
void
remove_inner_feats(Mat &_feat, Mat &_glabels)
{
	int i, k;
	Mat elem, _labero;
	uchar *labero;
	float *feat, *vs;
	
	CV_Assert(_feat.type() == CV_32FC1 && _feat.isContinuous()
		&& _feat.cols == NFEAT);
	CV_Assert(_glabels.type() == CV_32SC1 && _glabels.isContinuous());
	
	if(DEBUG){
		k = 0;
		for(i = 0; i < _feat.rows; i++){
			vs = (float*)_feat.ptr(i);
			if(!isnan(vs[FEAT_LAT])){
				k++;
			}
		}
		logprintf("number of feature before inner feats are removed: %d\n", k);
	}
	
	// erode clusters to remove borders from clusters
	elem = getStructuringElement(MORPH_RECT, Size(3, 3));
	erode(_glabels >= 0, _labero, elem);
	
	CV_Assert(_labero.type() == CV_8UC1 && _labero.isContinuous());
	
	// remove features if the pixel is in the eroded mask
	labero = _labero.data;
	feat = (float*)_feat.data;
	for(i = 0; i < (int)_glabels.total(); i++){
		if(labero[i]){
			for(k = 0; k < NFEAT; k++)
				feat[k] = NAN;
		}
		feat += NFEAT;
	}
}

// Update labels by nearest label training.
// _feat -- features (rows containing NaNs are removed)
// _lat, _lon, _sst, _delta -- latitude, longitude, SST, delta images
// _glabels -- global labels (output)
static void
nnlabel(Mat &_feat, const Mat &_lat, const Mat &_lon, const Mat &_sst, const Mat &_delta, Mat &_acspo, Mat _easyclouds, Mat &_glabels)
{
	int i, k, *indices, *glabels;
	float *vs, *vd, *lat, *lon, *sst, *delta;
	Mat _indices, _labdil;
	std::vector<float> q(NFEAT), dists(1);
	std::vector<int> ind(1);
	flann::SearchParams sparam(4);
	uchar *acspo, *easyclouds, *labdil;
	
	CV_Assert(_feat.type() == CV_32FC1 && _feat.isContinuous()
		&& _feat.cols == NFEAT);
	CV_Assert(_lat.type() == CV_32FC1 && _lat.isContinuous());
	CV_Assert(_lon.type() == CV_32FC1 && _lon.isContinuous());
	CV_Assert(_sst.type() == CV_32FC1 && _sst.isContinuous());
	CV_Assert(_delta.type() == CV_32FC1 && _delta.isContinuous());
	CV_Assert(_acspo.type() == CV_8UC1 && _acspo.isContinuous());
	CV_Assert(_glabels.type() == CV_32SC1 && _delta.isContinuous());
	CV_Assert(_easyclouds.type() == CV_8UC1 && _easyclouds.isContinuous());

	remove_inner_feats(_feat, _glabels);
	
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
	
	// dilate all the clusters
	dilate(_glabels >= 0, _labdil, getStructuringElement(MORPH_RECT, Size(7, 7)));
	CV_Assert(_labdil.type() == CV_8UC1 && _labdil.isContinuous());
	
	lat = (float*)_lat.data;
	lon = (float*)_lon.data;
	sst = (float*)_sst.data;
	delta = (float*)_delta.data;
	acspo = (uchar*)_acspo.data;
	glabels = (int*)_glabels.data;
	easyclouds = (uchar*)_easyclouds.data;
	labdil = (uchar*)_labdil.data;

	
	for(i = 0; i < (int)_sst.total(); i++){
		if(labdil[i] && glabels[i] < 0	// regions added by dilation
		&& easyclouds[i] == 0 && (acspo[i]&MaskCloud) != MaskCloudClear
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
logprintf("done searching nearest neighbors\n");
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

// Standard deviation filter, implemented as
//	dst = sqrt(blur(src^2) - blur(src)^2)
void
stdfilter(const Mat &src, Mat &dst, int ksize)
{
	int i;
	Mat b, bs, _tmp;
	float *tmp;
	
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
	Mat sst, reynolds, lat, lon, m14, m15, m16, elem, sstdil, sstero, rfilt, sstlap, medf, stdf, blurf;
	Mat m14gm, m15gm, m16gm;
	Mat acspo, gradmag, delta, omega, albedo, TQ, DQ, OQ, lut, glabels, feat, lam1, lam2, easyclouds, easyfronts;
	int ncid, n;
	char *path;
	Resample *r;

	if(argc != 2)
		eprintf("usage: %s granule\n", argv[0]);
	path = argv[1];
	logprintf("granule: %s\n", path);
	
	logprintf("reading and resampling...\n");
	r = new Resample;
	ncid = open_resampled(path, r);
	sst = readvar_resampled(ncid, r, "sst_regression");
	lat = readvar_resampled(ncid, r, "latitude");
	lon = readvar_resampled(ncid, r, "longitude");
	// TODO: interpolate acspo
	acspo = readvar_resampled(ncid, r, "acspo_mask");
	m14 = readvar_resampled(ncid, r, "brightness_temp_chM14");
	m15 = readvar_resampled(ncid, r, "brightness_temp_chM15");
	m16 = readvar_resampled(ncid, r, "brightness_temp_chM16");
	albedo = readvar_resampled(ncid, r, "albedo_chM7");
	n = nc_close(ncid);
	if(n != NC_NOERR)
		ncfatal(n, "nc_close failed for %s", path);

SAVENPY(lat);
SAVENPY(lon);
SAVENPY(acspo);
SAVENPY(sst);
SAVENPY(albedo);

	medianBlur(sst, medf, 5);
	stdfilter(sst-medf, stdf, 7);
	nanblur(sst, blurf, 7);
	easyclouds = (sst < SST_LOW) | (stdf > STD_THRESH)
		| (abs(sst - blurf) > EDGE_THRESH);
SAVENPY(easyclouds);
	
	logprintf("gradmag...\n");
	gradientmag(sst, gradmag);
SAVENPY(gradmag);

	logprintf("localmax...\n");
	localmax(gradmag, lam2, lam1, 1);
SAVENPY(lam2);

	easyfronts = (sst > SST_LOW) & (gradmag > 0.5)
		& (stdf < STD_THRESH) & (lam2 < -0.05);
SAVENPY(easyfronts);

	logprintf("delta...\n");
	delta = m15 - m16;
	omega = m14 - m15;
SAVENPY(delta);
SAVENPY(omega);

	logprintf("quantize sst delta...\n");
	quantize(lat, sst, delta, omega, gradmag, albedo, acspo, TQ, DQ, OQ, lut);
SAVENPY(TQ);
SAVENPY(DQ);
SAVENPY(OQ);
SAVENPY(lut);
//savenpy(savefilename(path, "_lut.npy"), lut);
	exit(0);

	logprintf("quantized featured...\n");
	quantized_features(TQ, DQ, lat, lon, sst, delta, glabels, feat);
SAVENPY(glabels);
SAVENPY(feat);

	nnlabel(feat, lat, lon, sst, delta, acspo, easyclouds, glabels);
savenpy("glabels_nn.npy", glabels);


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
	cmapimshow("gradmag", gradmag, COLORMAP_JET);

	while(waitKey(0) != 'q')
		;
*/

	logprintf("done\n");
	delete r;
	return 0;
}

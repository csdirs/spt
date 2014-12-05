#include "spt.h"


#define SCALE_LAT(x)	((x) * 10)
#define SCALE_LON(x)	((x) * 10)
#define SCALE_SST(x)	(x)
#define SCALE_DELTA(x)	((x) * 12)

// features
enum {
	FEAT_LAT,
	FEAT_LON,
	FEAT_SST,
	FEAT_DELTA,
	NFEAT,
};

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
	const float *sst, const float *delta, const float *omega, const float *lat, const float *lon,
	Mat &lut, Mat &_cclabels, Mat &_feat)
{
	Mat _mask, stats, centoids, _ccrename, _count, _avglat, _avgsst, _avgdelta, _avgomega;
	double *avglat, *avgsst, *avgdelta, *avgomega;
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
	ccrename[0] = COMP_INVALID;
	for(lab = 1; lab < ncc; lab++){
		if(stats.at<int>(lab, CC_STAT_AREA) >= 200)
			ccrename[lab] = newlab++;
		else
			ccrename[lab] = COMP_SPECKLE;
	}
	ncc = newlab;
	for(i = 0; i < size.area(); i++)
		cclabels[i] = ccrename[cclabels[i]];
	
	// remove these since they are wrong after the labels renaming
	stats.release();
	centoids.release();
	
	_count.create(ncc, 1, CV_32SC1);
	_avglat.create(ncc, 1, CV_64FC1);
	_avgsst.create(ncc, 1, CV_64FC1);
	_avgdelta.create(ncc, 1, CV_64FC1);
	_avgomega.create(ncc, 1, CV_64FC1);
	count = (int*)_count.data;
	avglat = (double*)_avglat.data;
	avgsst = (double*)_avgsst.data;
	avgdelta = (double*)_avgdelta.data;
	avgomega = (double*)_avgomega.data;
	memset(count, 0, sizeof(*count)*ncc);
	memset(avglat, 0, sizeof(*avglat)*ncc);
	memset(avgsst, 0, sizeof(*avgsst)*ncc);
	memset(avgdelta, 0, sizeof(*avgdelta)*ncc);
	memset(avgomega, 0, sizeof(*avgomega)*ncc);
	
	// compute average of lat, sst, delta, and omega per component
	for(i = 0; i < size.area(); i++){
		lab = cclabels[i];
		if(lab >= 0 && !isnan(sst[i]) && !isnan(delta[i])){
			avglat[lab] += lat[i];
			avgsst[lab] += sst[i];
			avgdelta[lab] += delta[i];
			avgomega[lab] += omega[i];
			count[lab]++;
		}
	}
	for(lab = 0; lab < ncc; lab++){
		avglat[lab] /= count[lab];
		avgsst[lab] /= count[lab];
		avgdelta[lab] /= count[lab];
		avgomega[lab] /= count[lab];
	}

	// query LUT, remove components that are cloud and rename labels to be contiguous.
	CV_Assert(lut.type() == CV_8SC1 && lut.isContinuous());
	memset(ccrename, 0, sizeof(*ccrename)*ncc);
	newlab = 0;
	for(lab = 0; lab < ncc; lab++){
		int idx[] = {
			quantize_lat(avglat[lab]),
			quantize_sst(avgsst[lab]),
			quantize_delta(avgdelta[lab]),
			quantize_omega(avgomega[lab]),
		};
		if(lut.at<char>(idx) != LUT_CLOUD)
			ccrename[lab] = newlab++;
		else
			ccrename[lab] = -1;
	}
	ncc = newlab;
	for(i = 0; i < size.area(); i++){
		lab = cclabels[i];
		if(lab >= 0)
			cclabels[i] = ccrename[lab];
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
	const Mat &_sst, const Mat &_delta, Mat &_omega, Mat &lut, Mat &_glabels, Mat &_feat)
{
	int i, glab, tqmax, dqmax, *glabels;
	float *lat, *lon, *sst, *delta, *omega, *feat;
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
	omega = (float*)_omega.data;
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
				sst, delta, omega, lat, lon, lut, _cclabels, _feat);
			CV_Assert(_cclabels.type() == CV_32SC1 && _cclabels.isContinuous());
			cclabels = (int*)_cclabels.data;
			
			#pragma omp critical
			if(ncc > 0){
				for(i = 0; i < (int)_cclabels.total(); i++){
					lab = cclabels[i];
					if(lab >= 0)
						glabels[i] = glab + lab;
					else if(lab == COMP_SPECKLE)
						glabels[i] = COMP_SPECKLE;
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
// _easyclouds -- easyclouds mask
// _gradmag -- gradient magnitude
// _glabels -- global labels (output)
static void
nnlabel(Mat &_feat, const Mat &_lat, const Mat &_lon, const Mat &_sst, const Mat &_delta,
	const Mat &_easyclouds, const Mat &_gradmag, Mat &_glabels)
{
	int i, k, *indices, *glabels;
	float *vs, *vd, *lat, *lon, *sst, *delta, *gradmag;
	Mat _indices, _labdil;
	std::vector<float> q(NFEAT), dists(1);
	std::vector<int> ind(1);
	flann::SearchParams sparam(4);
	uchar *easyclouds, *labdil;
	
	CV_Assert(_feat.type() == CV_32FC1 && _feat.isContinuous()
		&& _feat.cols == NFEAT);
	CV_Assert(_lat.type() == CV_32FC1 && _lat.isContinuous());
	CV_Assert(_lon.type() == CV_32FC1 && _lon.isContinuous());
	CV_Assert(_sst.type() == CV_32FC1 && _sst.isContinuous());
	CV_Assert(_delta.type() == CV_32FC1 && _delta.isContinuous());
	CV_Assert(_glabels.type() == CV_32SC1 && _delta.isContinuous());
	CV_Assert(_easyclouds.type() == CV_8UC1 && _easyclouds.isContinuous());
	CV_Assert(_gradmag.type() == CV_32FC1 && _gradmag.isContinuous());

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
	dilate(_glabels >= 0, _labdil, getStructuringElement(MORPH_RECT, Size(21, 21)));
	CV_Assert(_labdil.type() == CV_8UC1 && _labdil.isContinuous());
	
	lat = (float*)_lat.data;
	lon = (float*)_lon.data;
	sst = (float*)_sst.data;
	delta = (float*)_delta.data;
	glabels = (int*)_glabels.data;
	easyclouds = (uchar*)_easyclouds.data;
	gradmag = (float*)_gradmag.data;
	labdil = (uchar*)_labdil.data;

	
	for(i = 0; i < (int)_sst.total(); i++){
		if(labdil[i] && glabels[i] < 0	// regions added by dilation
		&& easyclouds[i] == 0
		&& !isnan(sst[i]) && !isnan(delta[i])
		&& (gradmag[i] > GRAD_LOW || glabels[i] == COMP_SPECKLE)
		&& SST_LOW < sst[i] && sst[i] < SST_HIGH
		&& DELTA_LOW < delta[i] && delta[i] < DELTA_HIGH){
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

	// TODO: erode glabels by 21, but not where gradmag < GRAD_LOW
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
SAVENPY(medf);
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

	logprintf("quantized featured...\n");
	quantized_features(TQ, DQ, lat, lon, sst, delta, omega, lut, glabels, feat);
SAVENPY(glabels);
SAVENPY(feat);

	nnlabel(feat, lat, lon, sst, delta, easyclouds, gradmag, glabels);
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

	cmapimshow("gradmag", gradmag, COLORMAP_JET);

	while(waitKey(0) != 'q')
		;
*/

	logprintf("done\n");
	delete r;
	return 0;
}

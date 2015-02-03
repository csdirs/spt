//
// SST Pattern Test
//

/*
TODO:
	Let C be the cluster
	Let R be the restored pixels within C
	Let F be the front

The size of restoration |R| should not be greater than |F|**2

*/


#include "spt.h"

#define CHECKMAT(M, T)	CV_Assert((M).type() == (T) && (M).isContinuous())

#define SCALE_LAT(x)	((x) * 10)
#define SCALE_LON(x)	((x) * 10)
#define SCALE_SST(x)	(x)
#define SCALE_DELTA(x)	(x)

// features
enum {
	FEAT_LAT,
	FEAT_LON,
	FEAT_SST,
	FEAT_DELTA,
	NFEAT,
};

// front statistics
enum {
	FSTAT_SIZE,	// front size in pixels
	FSTAT_LSIZE,	// left side size
	FSTAT_RSIZE,	// right side size
	FSTAT_SUMMAG,	// SST gradient magnitude sum
	FSTAT_OK,	// do we want this front?
	NFSTAT,
};

// thernal fronts and their sides
enum {
	FRONT_INVALID = -1,
	FRONT_INIT = 0,	// initial fronts (stage 1)
	FRONT_BIG,	// big enough fronts (stage 2)
	FRONT_OK,	// final fronts (stage 3)
	FRONT_LEFT,	// left side
	FRONT_RIGHT,	// right side
};

// Cloud mask values
enum {
	CMClear,
	CMProbably,
	CMSure,
	CMInvalid,
};

// Number of bits in cloud mask
enum {
	CMBits = 2,
};

enum {
	White	= 0xFFFFFF,
	Red		= 0xFF0000,
	Green	= 0x00FF00,
	Blue	= 0x0000FF,
	Yellow	= 0xFFFF00,
	JetRed	= 0x7F0000,
	JetBlue	= 0x00007F,
	JetGreen	= 0x7CFF79,
};

#define SetColor(v, c) do{ \
		(v)[0] = ((c)>>16) & 0xFF; \
		(v)[1] = ((c)>>8) & 0xFF; \
		(v)[2] = ((c)>>0) & 0xFF; \
	}while(0);

// Return a filename based on granule path path with suffix suf.
// e.g. savefilename("/foo/bar/qux.nc", ".png") returns "qux.png"
//
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

static void
diffcloudmask(const Mat &_old, const Mat &_new, Mat &_rgb)
{
	int i;
	uchar *old, *new1, *rgb, oval, nval;
	
	CHECKMAT(_old, CV_8UC1);
	CHECKMAT(_new, CV_8UC1);
	
	_rgb.create(_old.size(), CV_8UC3);
	rgb = _rgb.data;
	old = _old.data;
	new1 = _new.data;
	
	for(i = 0; i < (int)_old.total(); i++){
		oval = old[i]>>MaskCloudOffset;
		nval = new1[i] & 0x03;
		
		if(oval == CMProbably)
			oval = CMSure;
		if(nval == CMProbably)
			nval = CMSure;
		
		switch((oval<<CMBits) | nval){
		default:
			SetColor(rgb, Yellow);
			break;
		
		case (CMInvalid<<CMBits) | CMInvalid:
			SetColor(rgb, White);
			break;
		
		case (CMClear<<CMBits) | CMClear:
			SetColor(rgb, JetBlue);
			break;
		
		case (CMSure<<CMBits) | CMSure:
			SetColor(rgb, JetRed);
			break;
		
		case (CMSure<<CMBits) | CMClear:
		case (CMInvalid<<CMBits) | CMClear:
			SetColor(rgb, JetGreen);
			break;
		}
		rgb += 3;
	}
}

int
connectedComponentsWithLimit(const Mat &mask, int connectivity, int lim, Mat &_cclabels)
{
	Mat stats, centoids, _ccrename;
	int i, ncc, lab, newlab, *cclabels, *ccrename;
	
	ncc = connectedComponentsWithStats(mask, _cclabels, stats, centoids, connectivity, CV_32S);
	if(ncc <= 1)
		return 0;
	
	CHECKMAT(_cclabels, CV_32SC1);
	_ccrename.create(ncc, 1, CV_32SC1);
	cclabels = (int*)_cclabels.data;
	ccrename = (int*)_ccrename.data;
	
	// Remove small connected components and rename labels to be contiguous.
	// Also, set background label 0 (where mask is 0) to -1.
	newlab = 0;
	ccrename[0] = COMP_INVALID;
	for(lab = 1; lab < ncc; lab++){
		if(stats.at<int>(lab, CC_STAT_AREA) >= lim)
			ccrename[lab] = newlab++;
		else
			ccrename[lab] = COMP_SPECKLE;
	}
	ncc = newlab;
	for(i = 0; i < (int)mask.total(); i++)
		cclabels[i] = ccrename[cclabels[i]];

	return ncc;
}

// Run connected component for t == tq and a == aq, and save the features
// for the connected components in _feat. Returns the number of connected
// components labeled in _cclabels.
//
// size -- size of image
// t -- quantized SST value
// d -- quantized delta value
// tq, dq -- quantized SST, delta images
// sst, delta -- original SST, delta images
// lat, lon -- latitude, longitude images
// _cclabels -- label assigned to pixels where (t == tq && d == dq) (output)
// _feat -- features corresponding to _cclabels (output)
//
int
clusterbin(Size size, int t, int d, const short *tq, const short *dq,
	const float *sst, const float *delta,
	const float *lat, const float *lon,
	Mat &_cclabels, Mat &_feat)
{
	Mat _mask, _count, _avgsst, _avgdelta;
	double *avgsst, *avgdelta;
	float *feat_lat, *feat_lon, *feat_sst, *feat_delta;
	int i, ncc, lab, *cclabels, *count;
	uchar *mask;
	
	// create mask for (t, d)
	_mask.create(size, CV_8UC1);
	mask = (uchar*)_mask.data;
	for(i = 0; i < (int)_mask.total(); i++)
		mask[i] = tq[i] == t && dq[i] == d ? 255 : 0;
	
	ncc = connectedComponentsWithLimit(_mask, 4, 200, _cclabels);
	if(ncc <= 0)
		return 0;
	
	CHECKMAT(_cclabels, CV_32SC1);
	cclabels = (int*)_cclabels.data;
	
	_count = Mat::zeros(ncc, 1, CV_32SC1);
	_avgsst = Mat::zeros(ncc, 1, CV_64FC1);
	_avgdelta = Mat::zeros(ncc, 1, CV_64FC1);
	count = (int*)_count.data;
	avgsst = (double*)_avgsst.data;
	avgdelta = (double*)_avgdelta.data;
	
	// compute average per component
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

// Cluster and find features. Returns the number of clusters labeled in _glabels.
//
// TQ, DQ -- quantized SST and delta images
// _lat, _lon -- latitude, longitude images 
// _sst, _delta -- SST, delta images
// _glabels -- global labels (output)
// _feat -- features (output)
//
int
cluster(const Mat &TQ, const Mat &DQ, const Mat &_lat, const Mat &_lon,
	const Mat &_sst, const Mat &_delta,
	Mat &_glabels, Mat &_feat)
{
	int i, glab, *glabels;
	float *lat, *lon, *sst, *delta, *feat;
	short *tq, *dq;
	
	CHECKMAT(TQ, CV_16SC1);
	CHECKMAT(DQ, CV_16SC1);
	CHECKMAT(_lat, CV_32FC1);
	CHECKMAT(_lon, CV_32FC1);
	CHECKMAT(_sst, CV_32FC1);
	CHECKMAT(_delta, CV_32FC1);
	
	_glabels.create(_sst.size(), CV_32SC1);
	_feat.create(NFEAT, _sst.total(), CV_32FC1);
	
	tq = (short*)TQ.data;
	dq = (short*)DQ.data;
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
	for(int t = quantize_sst(SST_LOW); t < quantize_sst(SST_HIGH); t++){
		#pragma omp parallel for
		for(int d = quantize_delta(DELTA_LOW); d < quantize_delta(DELTA_HIGH); d++){
			Mat _cclabels;
			int ncc, lab, *cclabels;
			
			ncc = clusterbin(_sst.size(), t, d, tq, dq,
				sst, delta, lat, lon, _cclabels, _feat);
			CHECKMAT(_cclabels, CV_32SC1);
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
	return glab;
}

// Remove features from _feat that are not on the border of clusters defined
// by the clustering labels in _glabels.
//
void
remove_inner_feats(Mat &_feat, const Mat &_glabels)
{
	int i, k;
	Mat elem, _labero;
	uchar *labero;
	float *feat, *vs;
	
	CHECKMAT(_feat, CV_32FC1);
	CV_Assert(_feat.cols == NFEAT);
	CHECKMAT(_glabels, CV_32SC1);
	
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
	
	CHECKMAT(_labero, CV_8UC1);
	
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
//
// _feat -- features (rows containing NaNs are removed)
// _lat, _lon, _sst, _delta -- latitude, longitude, SST, delta images
// _easyclouds -- easyclouds mask
// _gradmag -- gradient magnitude
// _glabels -- global labels (input & output)
//
static void
nnlabel(Mat &_feat, const Mat &_lat, const Mat &_lon,
	const Mat &_sst, const Mat &_delta,
	const Mat &_easyclouds, const Mat &_gradmag, Mat &_glabels)
{
	int i, k, *indices, *glabels;
	float *vs, *vd, *lat, *lon, *sst, *delta, *gradmag;
	Mat _indices, _labdil;
	std::vector<float> q(NFEAT), dists(1);
	std::vector<int> ind(1);
	flann::SearchParams sparam(4);
	uchar *easyclouds, *labdil;
	
	CHECKMAT(_feat, CV_32FC1);
	CV_Assert(_feat.cols == NFEAT);
	CHECKMAT(_lat, CV_32FC1);
	CHECKMAT(_lon, CV_32FC1);
	CHECKMAT(_sst, CV_32FC1);
	CHECKMAT(_delta, CV_32FC1);
	CHECKMAT(_easyclouds, CV_8UC1);
	CHECKMAT(_gradmag, CV_32FC1);
	CHECKMAT(_glabels, CV_32SC1);

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
	CHECKMAT(_labdil, CV_8UC1);
	
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
			if(dists[0] < 5)
				glabels[i] = glabels[indices[ind[0]]];
		}
	}
logprintf("done searching nearest neighbors\n");

	// TODO: erode glabels by 21, but not where gradmag < GRAD_LOW
}

#define VAR_NAME	"spt_mask"
const char VAR_UNITS[] = "none";
const char VAR_DESCR[] = "SPT mask packed into 1 byte: bits1-2 (00=clear; 01=probably clear; 10=cloudy; 11=clear-sky mask undefined); bit3 (0=no thermal front; 1=thermal front)";

// Write spt into NetCDF dataset ncid as variable named "spt_mask".
//
void
write_spt_mask(int ncid, Mat &spt)
{
	int i, n, varid, ndims, dimids[2];
	nc_type xtype;
	size_t len;
	
	CHECKMAT(spt, CV_8UC1);
	
	// chunk sizes used by acspo_mask
	const size_t chunksizes[] = {1024, 3200};
	
	// It's not possible to delete a NetCDF variable, so attempt to use
	// the variable if it already exists. Create the variable if it does not exist.
	n = nc_inq_varid(ncid, VAR_NAME, &varid);
	if(n != NC_NOERR){
		n = nc_inq_dimid(ncid, "scan_lines_along_track", &dimids[0]);
		if(n != NC_NOERR)
			ncfatal(n, "nc_inq_dimid failed");

		n = nc_inq_dimid(ncid, "pixels_across_track", &dimids[1]);
		if(n != NC_NOERR)
			ncfatal(n, "nc_inq_dimid failed");
		
		n = nc_def_var(ncid, VAR_NAME, NC_UBYTE, nelem(dimids), dimids, &varid);
		if(n != NC_NOERR)
			ncfatal(n, "nc_def_var failed");
		n = nc_def_var_chunking(ncid, varid, NC_CHUNKED, chunksizes);
		if(n != NC_NOERR)
			ncfatal(n, "nc_def_var_chunking failed");
		n = nc_def_var_deflate(ncid, varid, 0, 1, 1);
		if(n != NC_NOERR)
			ncfatal(n, "setting deflate parameters failed");
		
		n = nc_put_att_text(ncid, varid, "UNITS", nelem(VAR_UNITS)-1, VAR_UNITS);
		if(n != NC_NOERR)
			ncfatal(n, "setting attribute UNITS failed");
		n = nc_put_att_text(ncid, varid, "Description", nelem(VAR_DESCR)-1, VAR_DESCR);
		if(n != NC_NOERR)
			ncfatal(n, "setting attribute Description failed");
	}
	
	// Varify that the netcdf variable has correct type and dimensions.
	n = nc_inq_var(ncid, varid, NULL, &xtype, &ndims, dimids, NULL);
	if(n != NC_NOERR)
		ncfatal(n, "nc_inq_var failed");
	if(xtype != NC_UBYTE)
		eprintf("variable type is %d, want %d\n", xtype, NC_UBYTE);
	if(ndims != 2)
		eprintf("variable dims is %d, want 2\n", ndims);
	for(i = 0; i < 2; i++){
		n = nc_inq_dimlen(ncid, dimids[i], &len);
		if(n != NC_NOERR)
			ncfatal(n, "nc_inq_dimlen failed");
		if(len != (size_t)spt.size[i])
			eprintf("dimension %d is %d, want %d\n", i, len, spt.size[i]);
	}
	
	// Write data into netcdf variable.
	n = nc_put_var_uchar(ncid, varid, spt.data);
	if(n != NC_NOERR)
		ncfatal(n, "nc_putvar_uchar failed");
}

void
compute_spt_mask(Mat &_acspo, Mat &_labels, Mat &_spt)
{
	int i;
	uchar *acspo, *labels, *spt, cm;
	
	CHECKMAT(_acspo, CV_8UC1);
	CHECKMAT(_labels, CV_8SC1);
	
	_spt.create(_acspo.size(), CV_8UC1);
	
	acspo = _acspo.data;
	labels = _labels.data;
	spt = _spt.data;
	
	for(i = 0; i < (int)_acspo.total(); i++){
		cm = acspo[i] & MaskCloud;
		if((cm == MaskCloudProbably || cm == MaskCloudSure) && labels >= 0)
			spt[i] = 0;
		else
			spt[i] = cm >> MaskCloudOffset;
	}
}

// Compute thermal fronts. Let:
//
//	f(x) = 1.0/(1+exp(100*(x+0.01)))
//	g(x) = 1.0/(1+exp(-30*(x-0.15)))
//	h(x) = 1.0/(1+exp(30*(x-0.15)))
//	prod1 = f(lam2)*g(gradmag)*h(stdf)
//	prod2 = f(lam2)*clip(gradmag, 0, 1)
//
// fronts must satisfy:
//	- not in dilated easy clouds
//	- over domain added by nearest neighbor search
//	- prod1 > 0.5 || prod2 > 0.5
//
// lam2 -- local max
// gradmag -- gradient magnitude
// stdf -- stdfilter(sst - medianBlur(sst))
// ncc -- number of connected components labeled in glabels
// glabels -- cluster labels before nearest neighbor lookup
// glabels_nn -- cluster labels after nearest neighbor lookup
// easyclouds -- easy clouds
// fronts -- thermal fronts (output)
//
void
thermal_fronts(const Mat &_lam2, const Mat &_gradmag, const Mat &_stdf,
	const Mat &_glabels, const Mat &_glabels_nn,
	const Mat &easyclouds, Mat &_fronts)
{
	Mat _dilc;
	float *lam2, *gradmag, *stdf;
	double m, llam, lmag, lstdf;
	int i, *glabels, *glabels_nn;
	uchar *dilc;
	schar *fronts;
	
	CHECKMAT(_lam2, CV_32FC1);
	CHECKMAT(_gradmag, CV_32FC1);
	CHECKMAT(_stdf, CV_32FC1);
	CHECKMAT(_glabels, CV_32SC1);
	CHECKMAT(_glabels_nn, CV_32SC1);
	CHECKMAT(easyclouds, CV_8UC1);
	_fronts.create(_glabels.size(), CV_8SC1);
	
	lam2 = (float*)_lam2.data;
	gradmag = (float*)_gradmag.data;
	stdf = (float*)_stdf.data;
	glabels = (int*)_glabels.data;
	glabels_nn = (int*)_glabels_nn.data;
	fronts = (schar*)_fronts.data;
	
	// dilate easyclouds
	dilate(easyclouds, _dilc, getStructuringElement(MORPH_RECT, Size(7, 7)));
	CHECKMAT(_dilc, CV_8UC1);
	dilc = _dilc.data;
	
	// compute thermal fronts image
	for(i = 0; i < (int)_glabels.total(); i++){
		fronts[i] = FRONT_INVALID;
		
		// continue if in (dilated) easyclouds
		// or not in domain added by nearest neighbor
		if(dilc[i] || glabels_nn[i] < 0 || glabels[i] >= 0)
			continue;
		
		// it's front if logit'(lam2) * clip(gradmag, 0, 1) > 0.5
		llam = 1.0/(1+exp(100*(lam2[i]+0.01)));
		m = gradmag[i];
		if(m > 1)
			m = 1;
		if(llam*m > 0.5){
			fronts[i] = FRONT_INIT;
			continue;
		}
		
		// it's front if logit'(lam2)*logit''(stdf)*logit'''(stdf) > 0.5
		lmag = 1.0/(1+exp(-30*(gradmag[i]-0.15)));
		lstdf = 1.0/(1+exp(30*(stdf[i]-0.15)));
		if(llam*lmag*lstdf > 0.5)
			fronts[i] = FRONT_INIT;
	}
}

// Narrow down the number of thermal fronts and find clusters that are
// adjacent to those fronts.
//
// _fronts -- fronts mask
// _dy -- column-wise gradient
// _dx -- row-wise gradient
// _gradmag -- gradient magnitude
// nclust -- number of clusters
// _clust -- clustering labels
// _acspo -- ACSPO mask
// alpha -- factor multiplied to gradient for obtaining the left/right sides of fronts
// _fronts -- fronts image (input & output)
// _adjclust -- mask indicated if the a cluster is adjacent to a front (output)
//
void
find_adjclust(const Mat &_dy, const Mat &_dx, const Mat &_gradmag,
	int nclust, const Mat &_clust, const Mat &_acspo, double alpha,
	Mat &_fronts, Mat &_adjclust)
{
	Mat _cclabels, _fstats;
	int i, j, *p, nfront, y, x, k, left, right, *cclabels, *clust;
	schar *fronts;
	float *dy, *dx, *gradmag;
	double dy1, dx1, *fstats, *fs, t;
	uchar *acspo, *adjclust;
	
	CHECKMAT(_dy, CV_32FC1);
	CHECKMAT(_dx, CV_32FC1);
	CHECKMAT(_gradmag, CV_32FC1);
	CHECKMAT(_clust, CV_32SC1);
	CHECKMAT(_acspo, CV_8UC1);
	CHECKMAT(_fronts, CV_8SC1);
	dy = (float*)_dy.data;
	dx = (float*)_dx.data;
	gradmag = (float*)_gradmag.data;
	clust = (int*)_clust.data;
	acspo = (uchar*)_acspo.data;
	fronts = (schar*)_fronts.data;
	
	// initialize output in case we bail early (e.g. if nfront <= 0)
	_adjclust = Mat::zeros(nclust, 1, CV_8UC1);
	
	// run connected components on fronts, eliminating small fronts
	nfront = connectedComponentsWithLimit(_fronts==FRONT_INIT, 8, 50, _cclabels);
	if(nfront <= 0)
		return;
	CHECKMAT(_cclabels, CV_32SC1);
	cclabels = (int*)_cclabels.data;
	logprintf("initial number of fronts: %d\n", nfront);
	
	int countsize[] = {nfront, nclust};
	SparseMat leftcount(nelem(countsize), countsize, CV_32SC1);
	SparseMat rightcount(nelem(countsize), countsize, CV_32SC1);

	_fstats = Mat::zeros(nfront, NFSTAT, CV_64FC1);
	fstats = (double*)_fstats.data;
	
	// find left and right sides of the fronts, and their statistics
	k = 0;
	for(y = 0; y < _fronts.rows; y++){
		for(x = 0; x < _fronts.cols; x++){
			if(cclabels[k] < 0){
				k++;
				continue;
			}
			
			// normalize vector (dy, dx) and multiply it by alpha
			dy1 = round(alpha * dy[k]/gradmag[k]);
			dx1 = round(alpha * dx[k]/gradmag[k]);
			
			// compute indices of left and right sides
			left = k + dx1*_fronts.cols - dy1;
			right = k - dx1*_fronts.cols + dy1;
			
			// compute statistics of front
			fs = &fstats[NFSTAT * cclabels[k]];
			fs[FSTAT_SIZE]++;
			fs[FSTAT_SUMMAG] += gradmag[k];
			fronts[k] = FRONT_BIG;
			if(0 <= left && left < (int)_fronts.total()
			&& (clust[left] >= 0 || (acspo[left]&MaskLand) != 0)){
				fs[FSTAT_LSIZE]++;
				fronts[left] = FRONT_LEFT;
				(*(int*)leftcount.ptr(cclabels[k], clust[left], true))++;
			}
			if(0 <= right && right < (int)_fronts.total()
			&& (clust[right] >= 0 || (acspo[right]&MaskLand) != 0)){
				fs[FSTAT_RSIZE]++;
				fronts[right] = FRONT_RIGHT;
				(*(int*)rightcount.ptr(cclabels[k], clust[right], true))++;
			}

			k++;
		}
	}
	
	logprintf("number of pixels left of fronts: %lu\n", leftcount.nzcount());
	logprintf("number of pixels right of fronts: %lu\n", rightcount.nzcount());
	
	adjclust = _adjclust.data;
	
	// find which clusters are adjacent to a front
	for(i = 0; i < nfront; i++){
		fs = &fstats[NFSTAT * i];
		t = 0.7*fs[FSTAT_SIZE];
		fs[FSTAT_OK] = fs[FSTAT_LSIZE] > t && fs[FSTAT_RSIZE] > t
			&& fs[FSTAT_SUMMAG]/fs[FSTAT_SIZE] > GRAD_THRESH;
		
		if(!fs[FSTAT_OK])
			continue;
		
		for(j = 0; j < nclust; j++){
			p = (int*)leftcount.ptr(i, j, false);
			if(p && *p/(double)fs[FSTAT_LSIZE] > 0.3)
				adjclust[j] = 255;

			p = (int*)rightcount.ptr(i, j, false);
			if(p && *p/(double)fs[FSTAT_RSIZE] > 0.3)
				adjclust[j] = 255;
		}
	}
	
	// set fronts that are accepted in fronts image
	for(k = 0; k < (int)_fronts.total(); k++){
		if(cclabels[k] < 0)
			continue;
		
		fs = &fstats[NFSTAT * cclabels[k]];
		if(fs[FSTAT_OK])
			fronts[k] = FRONT_OK;
	}
if(DEBUG)savenc("flabels.nc", _cclabels);
if(DEBUG)savenc("fstats.nc", _fstats);
}

// Resample ACSPO cloud mask to fill in deletion zones.
//
// _acspo -- ACSPO mask already sorted by latitude
// _acloud -- ACSPO cloud mask represented using float32
//	where NAN is land/ice/etc., 1 is "confidently cloudy", 0 is "clear sky",
//	and number in (0, 1) is result of interpolation on deletion zones near
//	cloud/clear sky boundary. (output)
//
void
resample_acloud(const Resample *r, const Mat &_acspo, Mat &_acloud)
{
	int i;
	uchar *acspo;
	float *acloud;
	
	CHECKMAT(_acspo, CV_8UC1);
	_acloud.create(_acspo.size(), CV_32FC1);
	acspo = _acspo.data;
	acloud = (float*)_acloud.data;
	
	// Prepare ACSPO cloud mask for resampling.
	// The resampling code only works with float32, and interpolates
	// over NAN values.
	for(i = 0; i < (int)_acspo.total(); i++){
		switch(acspo[i]&MaskCloud){
		default:
			acloud[i] = 0;
			break;
		case MaskCloudInvalid:
			acloud[i] = NAN;
			break;
		case MaskCloudSure:
			acloud[i] = 1;
			break;
		}
	}
	resample_float32(r, _acloud, _acloud, false);
}

// Create the SPT mask containing the cloud mask with new clear-sky restored
// and the fronts.
//
// r -- resampling context
// acspo -- ACSPO mask sorted by latitude
// _clust -- clustering labels image
// _adjclust -- for each cluster, indicates if it's adjacent to a thernal front
// _fronts -- fronts image
// _spt -- spt mask sorted by latitude (output)
//
void
get_spt(const Resample *r, const Mat &_acspo, const Mat &_clust,
	const Mat &_adjclust, const Mat &_fronts, Mat &_spt)
{
	Mat _labels, stats, centoids, _acloud, _mask;
	float *acloud;
	int i, *clust, *labels, nlab, n;
	uchar *adjclust, *mask, *acspo, *spt;
	schar *fronts;
	
	CHECKMAT(_acspo, CV_8UC1);
	CHECKMAT(_clust, CV_32SC1);
	CHECKMAT(_adjclust, CV_8UC1);
	CHECKMAT(_fronts, CV_8SC1);
	acspo = _acspo.data;
	clust = (int*)_clust.data;
	adjclust = _adjclust.data;
	fronts = (schar*)_fronts.data;
	
	resample_acloud(r, _acspo, _acloud);
	CHECKMAT(_acloud, CV_32FC1);
if(DEBUG)savenc("acloud.nc", _acloud);
	acloud = (float*)_acloud.data;
	
	// Create a mask containing the new clear-sky pixels that we may potentially
	// restore in ACSPO. It consists of pixels that satisfy:
	// - belongs to a cluster adjacent to a front
	// - ACSPO says it's confidently cloudy
	// We're using the resampled/interpolated ACSPO cloud mask (acloud)
	// so that the deleted zones doesn't split clusters into smaller ones.
	_mask.create(_acspo.size(), CV_8UC1);
	mask = _mask.data;
	for(i = 0; i < (int)_mask.total(); i++){
		mask[i] = 0;
		if(clust[i] >= 0 && adjclust[clust[i]] && !isnan(acloud[i]) && acloud[i] > 0)
			mask[i] = 255;
	}
	
	// run connected components on "new clear-sky" mask
	nlab = connectedComponentsWithStats(_mask, _labels, stats, centoids, 4, CV_32S);
	if(nlab <= 1)
		return;
	CHECKMAT(_labels, CV_32SC1);
	labels = (int*)_labels.data;
	
	// Create spt mask containing the fronts and the cloud mask.
	// We disable small components from being restored as clear-sky.
	_spt.create(_acspo.size(), CV_8UC1);
	spt = _spt.data;
	for(i = 0; i < (int)_labels.total(); i++){
		spt[i] = acspo[i] >> MaskCloudOffset;
		n = labels[i];
		if(n > 0 && stats.at<int>(n, CC_STAT_AREA) >= 100
		&& (acspo[i]&MaskCloud) == MaskCloudSure)
			spt[i] = MaskCloudClear >> MaskCloudOffset;

		if(fronts[i] == FRONT_OK)
			spt[i] |= 0x04;
	}
}

int
main(int argc, char **argv)
{
	Mat sst, cmc, anomaly, lat, lon, m14, m15, m16, medf, stdf, blurf,
		acspo, acspo1, dX, dY, gradmag, delta, omega, albedo, TQ, DQ, OQ, AQ,
		lut, glabels, glabels_nn, feat, lam1, lam2,
		easyclouds, easyfronts, fronts, adjclust, spt, spt1, diff;
	int ncid, n, nclust;
	char *path;
	Resample *r;

	if(argc != 2)
		eprintf("usage: %s granule\n", argv[0]);
	path = argv[1];
	logprintf("granule: %s\n", path);
	
	logprintf("reading and resampling...\n");
	r = new Resample;
	ncid = open_resampled(path, r, NC_WRITE);
	sst = readvar_resampled(ncid, r, "sst_regression");
	cmc = readvar_resampled(ncid, r, "sst_reynolds");
	lat = readvar_resampled(ncid, r, "latitude");
	lon = readvar_resampled(ncid, r, "longitude");
	acspo = readvar_resampled(ncid, r, "acspo_mask");
	m14 = readvar_resampled(ncid, r, "brightness_temp_chM14");
	m15 = readvar_resampled(ncid, r, "brightness_temp_chM15");
	m16 = readvar_resampled(ncid, r, "brightness_temp_chM16");
	albedo = readvar_resampled(ncid, r, "albedo_chM7");

SAVENC(lat);
SAVENC(acspo);
SAVENC(sst);
SAVENC(albedo);

	logprintf("computing gradmag, etc....\n");
	delta = m15 - m16;
	omega = m14 - m15;
	anomaly = sst - cmc;
	gradientmag(sst, dX, dY, gradmag);
	localmax(gradmag, lam2, lam1, 1);
SAVENC(m15);
SAVENC(m16);
SAVENC(delta);
SAVENC(omega);
SAVENC(anomaly);
SAVENC(gradmag);
SAVENC(lam2);

	medianBlur(sst, medf, 5);
	stdfilter(sst-medf, stdf, 7);
	//nanblur(sst, blurf, 7);
	easyclouds = (sst < SST_LOW) | (stdf > STD_THRESH);
		//| (abs(sst - blurf) > EDGE_THRESH);
SAVENC(medf);
SAVENC(stdf);
SAVENC(easyclouds);

	//easyfronts = (sst > SST_LOW) & (gradmag > 0.5)
	//	& (stdf < STD_THRESH) & (lam2 < -0.05);

	logprintf("quantizing sst delta...\n");
	quantize(lat, sst, delta, omega, anomaly, gradmag, stdf, albedo, acspo, TQ, DQ, OQ, AQ, lut);

	logprintf("computing quantized features...\n");
	nclust = cluster(TQ, DQ, lat, lon, sst, delta, glabels, feat);
SAVENC(glabels);

	glabels_nn = glabels.clone();
	nnlabel(feat, lat, lon, sst, delta, easyclouds, gradmag, glabels_nn);
SAVENC(glabels_nn);
	
	logprintf("finding thermal fronts...\n");
	thermal_fronts(lam2, gradmag, stdf, glabels, glabels_nn, easyclouds, fronts);

	logprintf("finding clusters adjacent to fronts...\n");
	find_adjclust(dY, dX, gradmag, nclust, glabels_nn, acspo, 5, fronts, adjclust);
SAVENC(fronts);
SAVENC(adjclust);

	logprintf("creating spt mask...\n");
	get_spt(r, acspo, glabels_nn, adjclust, fronts, spt);
SAVENC(spt);

	logprintf("saving spt mask...\n");
	spt1 = resample_unsort(r->sind, spt);
	write_spt_mask(ncid, spt1);
	
	acspo1 = resample_unsort(r->sind, acspo);
	diffcloudmask(acspo1, spt1, diff);
	resize(diff, diff, Size(), 1/6.0, 1/6.0, INTER_AREA);
	imwrite("diff.png", diff);
SAVENC(diff);

	n = nc_close(ncid);
	if(n != NC_NOERR)
		ncfatal(n, "nc_close failed for %s", path);

	delete r;
	logprintf("done\n");
	return 0;
}

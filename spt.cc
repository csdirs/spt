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
#include "fastBilateral.hpp"

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

class Var	// variable
{
public:
	bool inrange(float val) {
		return !isnan(val) && min <= val && val <= max;
	};
	virtual int quantize(float val) = 0;
	Mat mat;
	float min, max;
	double scalefeat;	// scale feature by
	bool avgfeat;	// use average for a cluster as feature
};

class SST : public Var
{
public:
	SST(Mat &m) {
		mat = m;
		min = SST_LOW;
		max = SST_HIGH;
		avgfeat = true;
		scalefeat = 1.0;
	};
	int quantize(float val) {
		return cvRound((val - SST_LOW) * (1.0/TQ_STEP));
	};
};

class Delta : public Var
{
public:
	Delta(Mat &m) {
		mat = m;
		min = DELTA_LOW;
		max = DELTA_HIGH;
		avgfeat = true;
		scalefeat = 5.0;
	};
	int quantize(float val) {
		return cvRound((val - DELTA_LOW) * (1.0/DQ_STEP));
	};
};

class CMCAnom : public Var
{
public:
	CMCAnom(Mat &m) {
		mat = m;
		min = ANOMALY_LOW;
		max = ANOMALY_HIGH;
		avgfeat = true;
		scalefeat = 1.0;
	};
	int quantize(float val) {
		return cvRound((val - ANOMALY_LOW) * (1.0/AQ_STEP));
	};
};

class BilAnom : public Var
{
public:
	BilAnom(Mat &m) {
		mat = m;
		min = -999;
		max = 999;
		avgfeat = true;
		scalefeat = 1.0;
	};
	int quantize(float val) {
		return val <= 0 ? 0 : 1;
	};
};


class Lat : public Var
{
public:
	Lat(Mat &m) {
		mat = m;
		min = -90;
		max = 90;
		avgfeat = false;
		scalefeat = 10.0;
	};
	int quantize(float val) {
		float la = abs(val);
		if(la < 30)
			return 0;
		if(la < 45)
			return 1;
		if(la < 60)
			return 2;
		return 3;
	};
};

class Lon : public Var
{
public:
	Lon(Mat &m) {
		mat = m;
		min = -180;
		max = 180;
		avgfeat = false;
		scalefeat = 10.0;
	}
	int quantize(float val) { abort(); }
};

class QVar	// quantized variable
{
public:
	QVar(Mat &m) { mat = m; };
	Mat mat;
	int min, max;
};


// Quantize variables.
//
// n -- number of variables
// src -- images of variables (source) that needs to be quantized
// _omega, _sstmag -- omega and gradient magnitude image
// _deltamag -- delta image
// dst -- destination where quantized images are stored (output)
//
static void
quantize(int n, Var **src, const Mat &_omega, const Mat &_sstmag,
	const Mat &_deltamag, QVar **dst)
{
	int i, k;
	bool ok;
	float *omega, *gm, *deltamag;
	
	CV_Assert(n > 0);
	Size size = src[0]->mat.size();
	
	for(k = 0; k < n; k++)
		CHECKMAT(src[k]->mat, CV_32FC1);
	for(k = 0; k < n; k++){
		dst[k]->mat.create(size, CV_16SC1);
		dst[k]->mat = Scalar(-1);
	}
	
	CHECKMAT(_omega, CV_32FC1);
	CHECKMAT(_sstmag, CV_32FC1);
	CHECKMAT(_deltamag, CV_32FC1);
	omega = (float*)_omega.data;
	gm = (float*)_sstmag.data;
	deltamag = (float*)_deltamag.data;
	
	// quantize variables
	for(i = 0; i < (int)size.area(); i++){
		if(gm[i] > GRAD_LOW		// || delta[i] < -0.5
		|| deltamag[i] > DELTAMAG_LOW
		|| (omega[i] < OMEGA_LOW || omega[i] > OMEGA_HIGH))
			continue;
		
		ok = true;
		for(k = 0; k < n; k++){
			float *s = (float*)src[k]->mat.data;
			if(!src[k]->inrange(s[i])){
				ok = false;
				break;
			}
		}
		if(ok){
			for(k = 0; k < n; k++){
				float *s = (float*)src[k]->mat.data;
				short *d = (short*)dst[k]->mat.data;
				d[i] = src[k]->quantize(s[i]);
			}
		}
	}
	
	for(k = 0; k < n; k++){
		dst[k]->min = src[k]->quantize(src[k]->min);
		dst[k]->max = src[k]->quantize(src[k]->max);
	}
}


// Return a filename based on granule path path with suffix suf.
// e.g. savefilename("/foo/bar/qux.nc", ".png") returns "qux.png"
//
static char*
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

// Compute RGB diff image of cloud mask.
//
// _old -- old cloud mask (usually ACSPO cloud mask)
// _new -- new cloud mask (usually SPT cloud mask)
// _rgb -- RGB diff image (output)
//
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

// Connected components wrapper that limits the minimum size of components.
//
// mask -- the image to be labeled
// connectivity -- 8 or 4 for 8-way or 4-way connectivity respectively
// lim -- limit on the minimum size of components
// _cclabels -- destination labeled image (output)
//
static int
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

// Run connected component for v1 == q1 and v2 == q2, and save the features
// for the connected components in feat. Returns the number of connected
// components labeled in _cclabels.
//
// size -- size of image
// v1, v2 -- quantized values
// q1, q2 -- quantized images
// _vars -- variables used as features
// _cclabels -- label assigned to pixels where (v1 == q1 && v2 == q2) (output)
// feat -- features corresponding to _cclabels (output)
//
static int
clusterbin(Size size, int v1, int v2, const short *q1, const short *q2,
	Var **_vars, Mat &_cclabels, float *feat)
{
	Mat _mask, _count, _avg[NFEAT];
	double *avg[NFEAT];
	float *vars[NFEAT];
	int i, ncc, lab, *cclabels, *count;
	uchar *mask;
	
	// create mask for (v1, v2) == (q1, q2)
	_mask.create(size, CV_8UC1);
	mask = (uchar*)_mask.data;
	for(i = 0; i < (int)_mask.total(); i++)
		mask[i] = q1[i] == v1 && q2[i] == v2 ? 255 : 0;
	
	// run connected components on the mask
	ncc = connectedComponentsWithLimit(_mask, 4, 200, _cclabels);
	if(ncc <= 0)
		return 0;
	CHECKMAT(_cclabels, CV_32SC1);
	cclabels = (int*)_cclabels.data;
	
	// allocate temporary matrices for computing average per component
	_count = Mat::zeros(ncc, 1, CV_32SC1);
	count = (int*)_count.data;
	for(int k = 0; k < NFEAT; k++){
		vars[k] = (float*)_vars[k]->mat.data;
		avg[k] = NULL;
		if(_vars[k]->avgfeat){
			_avg[k] = Mat::zeros(ncc, 1, CV_64FC1);
			avg[k] = (double*)_avg[k].data;
		}
	}
	
	// compute average per component
	for(i = 0; i < size.area(); i++){
		lab = cclabels[i];
		if(lab < 0)
			continue;
		bool ok = true;
		for(int k = 0; k < NFEAT; k++){
			if(avg[k] && isnan(vars[k][i])){
				ok = false;
				break;
			}
		}
		if(ok){
			for(int k = 0; k < NFEAT; k++){
				if(avg[k])
					avg[k][lab] += vars[k][i];
			}
			count[lab]++;
		}
	}
	for(lab = 0; lab < ncc; lab++){
		for(int k = 0; k < NFEAT; k++){
			if(avg[k])
				avg[k][lab] /= count[lab];
		}
	}

	// compute features for each variables
	for(i = 0; i < size.area(); i++){
		lab = cclabels[i];
		if(lab >= 0){
			for(int k = 0; k < NFEAT; k++){
				double scale = _vars[k]->scalefeat;
				if(avg[k])
					feat[k] = scale * avg[k][lab];
				else
					feat[k] = scale * vars[k][i];
			}
		}
		feat += NFEAT;
	}
	return ncc;
}

// Cluster and find features. Returns the number of clusters labeled in _glabels.
//
// Q1, Q2 -- quantized variables
// vars -- variables used as features
// _glabels -- global labels (output)
// _feat -- features (output)
//
static int
cluster(QVar *Q1, QVar *Q2, Var **vars, Mat &_glabels, Mat &_feat)
{
	int i, glab, *glabels;
	float *feat;
	short *q1, *q2;
	
	CHECKMAT(Q1->mat, CV_16SC1);
	CHECKMAT(Q2->mat, CV_16SC1);
	for(int k = 0; k < NFEAT; k++)
		CHECKMAT(vars[k]->mat, CV_32FC1);
	
	Size size = vars[0]->mat.size();
	_glabels.create(size, CV_32SC1);
	_feat.create(size.area(), NFEAT, CV_32FC1);
	
	q1 = (short*)Q1->mat.data;
	q2 = (short*)Q2->mat.data;
	feat = (float*)_feat.data;
	glabels = (int*)_glabels.data;
	
	for(i = 0; i < (int)_feat.total(); i++)
		feat[i] = NAN;
	for(i = 0; i < (int)_glabels.total(); i++)
		glabels[i] = -1;
	
	glab = 0;
	#pragma omp parallel for
	for(int v1 = Q1->min; v1 <= Q1->max; v1++){
		#pragma omp parallel for
		for(int v2 = Q2->min; v2 <= Q2->max; v2++){
			Mat _cclabels;
			int ncc, lab, *cclabels;
			
			ncc = clusterbin(size, v1, v2, q1, q2, vars, _cclabels, feat);
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
	return glab;
}

// Remove features from _feat that are not on the border of clusters defined
// by the clustering labels in _glabels.
//
static void
removeinnerfeats(Mat &_feat, const Mat &_glabels)
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
		logprintf("labelnbrs: number of feature before inner feats are removed: %d\n", k);
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
// _var -- varibles used to query for nearest neighbor
// _easyclouds -- easyclouds mask
// _sstmag -- gradient magnitude
// _glabels -- global labels (input & output)
//
static void
labelnbrs(Mat &_feat, Var **_vars, const Mat &_easyclouds,
	const Mat &_sstmag, Mat &_glabels)
{
	int i, k, *indices, *glabels;
	float *vs, *vd, *vars[NFEAT], *sstmag;
	Mat _indices, _labdil;
	std::vector<float> q(NFEAT), dists(1);
	std::vector<int> ind(1);
	flann::SearchParams sparam(4);
	uchar *easyclouds, *labdil;
	
	CHECKMAT(_feat, CV_32FC1);
	CV_Assert(_feat.cols == NFEAT);
	CHECKMAT(_easyclouds, CV_8UC1);
	CHECKMAT(_sstmag, CV_32FC1);
	CHECKMAT(_glabels, CV_32SC1);
	for(k = 0; k < NFEAT; k++)
		CHECKMAT(_vars[k]->mat, CV_32FC1);

	removeinnerfeats(_feat, _glabels);
	
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
	logprintf("labelnbrs: reduced number of features: %d\n", k);
	
	logprintf("labelnbrs: building nearest neighbor indices...\n");
	flann::Index idx(_feat, flann::KMeansIndexParams(16, 1));
	logprintf("labelnbrs: searching nearest neighbor indices...\n");
	
	// dilate all the clusters
	dilate(_glabels >= 0, _labdil, getStructuringElement(MORPH_RECT, Size(21, 21)));
	CHECKMAT(_labdil, CV_8UC1);
	
	glabels = (int*)_glabels.data;
	easyclouds = (uchar*)_easyclouds.data;
	sstmag = (float*)_sstmag.data;
	labdil = (uchar*)_labdil.data;
	for(k = 0; k < NFEAT; k++)
		vars[k] = (float*)_vars[k]->mat.data;
	Size size = _vars[0]->mat.size();
	
	// label based on nearest neighbor
	for(i = 0; i < (int)size.area(); i++){
		if(!labdil[i] || glabels[i] >= 0	// not regions added by dilation
		|| easyclouds[i]
		|| (sstmag[i] < GRAD_LOW && glabels[i] != COMP_SPECKLE))
			continue;
		
		bool ok = true;
		for(k = 0; k < NFEAT; k++){
			if(!_vars[k]->inrange(vars[k][i])){
				ok = false;
				break;
			}
		}
		if(ok){
			for(k = 0; k < NFEAT; k++){
				q[k] = _vars[k]->scalefeat * vars[k][i];
			}
			idx.knnSearch(q, ind, dists, 1, sparam);
			if(dists[0] < 5)
				glabels[i] = glabels[indices[ind[0]]];
		}
	}
	logprintf("labelnbrs: done searching nearest neighbors\n");

	// TODO: erode glabels by 21, but not where sstmag < GRAD_LOW
}

#define VAR_NAME	"spt_mask"
const char VAR_UNITS[] = "none";
const char VAR_DESCR[] = "SPT mask packed into 1 byte: bits1-2 (00=clear; 01=probably clear; 10=cloudy; 11=clear-sky mask undefined); bit3 (0=no thermal front; 1=thermal front)";

// Write spt into NetCDF dataset ncid as variable named "spt_mask".
//
static void
writespt(int ncid, const Mat &spt)
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

// Find thermal fronts. Let:
//
//	f(x) = 1.0/(1+exp(100*(x+0.01)))
//	g(x) = 1.0/(1+exp(-30*(x-0.15)))
//	h(x) = 1.0/(1+exp(30*(x-0.15)))
//	q(x) = 1.0/(1+exp(100*(x-0.05)))
//	prod1 = f(lam2)*g(sstmag)*h(stdf)
//	prod2 = f(lam2)*clip(sstmag, 0, 1)
//
// fronts must satisfy:
//	- not in dilated easy clouds
//	- over domain added by nearest neighbor search
//	- prod1 > 0.5 || prod2 > 0.5
//
// lam2 -- local max
// sstmag -- gradient magnitude
// stdf -- stdfilter(sst - medianBlur(sst))
// ncc -- number of connected components labeled in glabels
// glabels -- cluster labels before nearest neighbor lookup
// glabelsnn -- cluster labels after nearest neighbor lookup
// easyclouds -- easy clouds
// fronts -- thermal fronts (output)
//
static void
findfronts(const Mat &_lam2, const Mat &_sstmag, const Mat &_stdf,
	const Mat &_deltamag, const Mat &_glabels, const Mat &_glabelsnn,
	const Mat &easyclouds, const Mat &_apm, Mat &_fronts)
{
	Mat _dilc, _dilq;
	float *lam2, *sstmag, *stdf, *deltamag;
	double m, llam, lmag, lstdf, ldel;
	int i, *glabels, *glabelsnn;
	uchar *dilc, *apm;
	schar *fronts;
	
	CHECKMAT(_lam2, CV_32FC1);
	CHECKMAT(_sstmag, CV_32FC1);
	CHECKMAT(_stdf, CV_32FC1);
	CHECKMAT(_deltamag, CV_32FC1);
	CHECKMAT(_glabels, CV_32SC1);
	CHECKMAT(_glabelsnn, CV_32SC1);
	CHECKMAT(easyclouds, CV_8UC1);
	CHECKMAT(_apm, CV_8UC1);
	_fronts.create(_glabels.size(), CV_8SC1);
	
	lam2 = (float*)_lam2.data;
	sstmag = (float*)_sstmag.data;
	stdf = (float*)_stdf.data;
	deltamag = (float*)_deltamag.data;
	glabels = (int*)_glabels.data;
	glabelsnn = (int*)_glabelsnn.data;
	fronts = (schar*)_fronts.data;
	apm = _apm.data;
	
	// dilate easyclouds
	dilate(easyclouds, _dilc, getStructuringElement(MORPH_RECT, Size(7, 7)));
	CHECKMAT(_dilc, CV_8UC1);
	dilc = _dilc.data;
/*	
	float *_dilq = 100*(_deltamag - 0.05);
	exp(_dilq, _dilq);
	erode(1.0/(1+_dilq) > 0.5, _dilq, getStructuringElement(MORPH_RECT, Size(7, 7)));
	if(DEBUG) savenc("dilq.nc", _dilq);
	CHECKMAT(_dilq, CV_32FC1);
	dilq = (float*)_dilq.data;
*/
	// compute thermal fronts image
	for(i = 0; i < (int)_glabels.total(); i++){
		fronts[i] = FRONT_INVALID;
		
		// continue if in (dilated) easyclouds
		// or not in domain added by nearest neighbor
		if(dilc[i] || glabelsnn[i] < 0 || glabels[i] >= 0)
			continue;

		if(apm[i] && sstmag[i] > 0.1 && sstmag[i]/deltamag[i] > 10){
			fronts[i] = FRONT_INIT;
			continue;
		}
		
		// it's front if logit'(lam2) * clip(sstmag, 0, 1) > 0.5
		llam = 1.0/(1+exp(100*(lam2[i]+0.01)));
		ldel = 1.0/(1+exp(100*(deltamag[i]-0.05)));
		m = sstmag[i];
		if(m > 1){
			m = 1;
		}
		if(llam*ldel*m > 0.5){
			fronts[i] = FRONT_INIT;
			continue;
		}
		
		// it's front if logit'(lam2)*logit''(stdf)*logit'''(stdf) > 0.5
		lmag = 1.0/(1+exp(-30*(sstmag[i]-0.15)));
		lstdf = 1.0/(1+exp(30*(stdf[i]-0.15)));
		ldel = 1.0/(1+exp(100*(deltamag[i]-0.1)));
		if(llam*lmag*lstdf*ldel > 0.5)
			fronts[i] = FRONT_INIT;
	}
}

// Narrow down the number of thermal fronts and find clusters that are
// adjacent to those fronts.
//
// _fronts -- fronts mask
// _dy -- column-wise gradient
// _dx -- row-wise gradient
// _sstmag -- gradient magnitude
// nclust -- number of clusters
// _clust -- clustering labels
// _acspo -- ACSPO mask
// alpha -- factor multiplied to gradient for obtaining the left/right sides of fronts
// _fronts -- fronts image (input & output)
// _adjclust -- mask indicated if the a cluster is adjacent to a front (output)
//
static void
findadjacent(const Mat &_dy, const Mat &_dx, const Mat &_sstmag,
	int nclust, const Mat &_clust, const Mat &_acspo, double alpha,
	Mat &_fronts, Mat &_adjclust)
{
	Mat _cclabels, _fstats;
	int i, j, *p, nfront, y, x, k, left, right, *cclabels, *clust;
	schar *fronts;
	float *dy, *dx, *sstmag;
	double dy1, dx1, *fstats, *fs, t;
	uchar *acspo, *adjclust;
	
	CHECKMAT(_dy, CV_32FC1);
	CHECKMAT(_dx, CV_32FC1);
	CHECKMAT(_sstmag, CV_32FC1);
	CHECKMAT(_clust, CV_32SC1);
	CHECKMAT(_acspo, CV_8UC1);
	CHECKMAT(_fronts, CV_8SC1);
	dy = (float*)_dy.data;
	dx = (float*)_dx.data;
	sstmag = (float*)_sstmag.data;
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
	logprintf("findadjacent: initial number of fronts: %d\n", nfront);
	
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
			dy1 = round(alpha * dy[k]/sstmag[k]);
			dx1 = round(alpha * dx[k]/sstmag[k]);
			
			// compute indices of left and right sides
			left = k + dx1*_fronts.cols - dy1;
			right = k - dx1*_fronts.cols + dy1;
			
			// compute statistics of front
			fs = &fstats[NFSTAT * cclabels[k]];
			fs[FSTAT_SIZE]++;
			fs[FSTAT_SUMMAG] += sstmag[k];
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
	
	logprintf("findadjacent: number of pixels left of fronts: %lu\n", leftcount.nzcount());
	logprintf("findadjacent: number of pixels right of fronts: %lu\n", rightcount.nzcount());
	
	adjclust = _adjclust.data;
	
	// find which clusters are adjacent to a front
	for(i = 0; i < nfront; i++){
		fs = &fstats[NFSTAT * i];
		t = 0.7*fs[FSTAT_SIZE];
		fs[FSTAT_OK] = fs[FSTAT_LSIZE] > t && fs[FSTAT_RSIZE] > t;
			//&& fs[FSTAT_SUMMAG]/fs[FSTAT_SIZE] > GRAD_THRESH;
		
		if(!fs[FSTAT_OK])
			continue;
		
		for(j = 0; j < nclust; j++){
			bool inleft = false;
			bool inright = false;
			
			p = (int*)leftcount.ptr(i, j, false);
			if(p && *p/(double)fs[FSTAT_LSIZE] > 0.3)
				inleft = true;

			p = (int*)rightcount.ptr(i, j, false);
			if(p && *p/(double)fs[FSTAT_RSIZE] > 0.3)
				inright = true;
			
			if((inleft && !inright) || (!inleft && inright))
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
}

// Resample ACSPO cloud mask to fill in deletion zones.
//
// _acspo -- ACSPO mask already sorted by latitude
// _acloud -- ACSPO cloud mask represented using float32
//	where NAN is land/ice/etc., 1 is "confidently cloudy", 0 is "clear sky",
//	and number in (0, 1) is result of interpolation on deletion zones near
//	cloud/clear sky boundary. (output)
//
static void
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
static void
getspt(const Resample *r, const Mat &_acspo, const Mat &_clust,
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
	acloud = (float*)_acloud.data;
	
	// Copy acspo cloud mask and fronts into spt mask
	_spt.create(_acspo.size(), CV_8UC1);
	spt = _spt.data;
	for(i = 0; i < (int)_acspo.total(); i++){
		spt[i] = acspo[i] >> MaskCloudOffset;
		if(fronts[i] == FRONT_OK)
			spt[i] |= 0x04;
	}
	
	// Create a mask containing the new clear-sky pixels that we may potentially
	// restore in ACSPO. It consists of pixels that satisfy:
	// - belongs to a cluster adjacent to a front
	// - ACSPO says it's confidently cloudy
	// We're using the resampled/interpolated ACSPO cloud mask (acloud)
	// so that the deleted zones doesn't split clusters into smaller ones.
	_mask.create(_acspo.size(), CV_8UC1);
	mask = _mask.data;
	for(i = 0; i < (int)_acspo.total(); i++){
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
	
	// Restored some clear-sky in spt mask.
	// We disable small components from being restored.
	for(i = 0; i < (int)_acspo.total(); i++){
		n = labels[i];
		if(n > 0 && stats.at<int>(n, CC_STAT_AREA) >= 100
		&& (acspo[i]&MaskCloud) == MaskCloudSure)
			spt[i] = MaskCloudClear >> MaskCloudOffset;
	}
}

// Bilateral filter wrapper.
static void
bilateral(const Mat &_sst, const Mat &_easyclouds, Mat &_dst, double sigma_color, double sigma_space)
{
	int i;
	Mat _src;
	float *sst, *src, *dst;
	uchar *easyclouds;

	CHECKMAT(_easyclouds, CV_8UC1);
	CHECKMAT(_sst, CV_32FC1);
	_src = _sst.clone();	// TODO: copy necessary here?
	
	easyclouds = _easyclouds.data;
	sst = (float*)_sst.data;
	src = (float*)_src.data;
	
	for(i = 0; i < (int)_sst.total(); i++){
		if(easyclouds[i])
			src[i] = -1;
		else if(sst[i] > SST_HIGH)
			src[i] = SST_HIGH;
	}
	// TODO: check if OpenCV's bilateralFilter is fast enough for us
	// and can replace this function.
	cv_extend::bilateralFilter(_src, _dst, sigma_color, sigma_space);

	CHECKMAT(_dst, CV_32FC1);
	dst = (float*)_dst.data;
	for(i = 0; i < (int)_sst.total(); i++){
		if(easyclouds[i])
			dst[i] = NAN;
	}
}

// Plus minus filter, which finds the edges between negative and positive
// values of source image.
// 
// _src -- source image
// _dst -- destination mask image (output)
//
static void
plusminus(const Mat &_src, Mat &_dst)
{
	Mat _tmp;
	float *src, *tmp;
	
	CHECKMAT(_src, CV_32FC1);
	_tmp.create(_src.size(), CV_32FC1);
	
	enum {
		wsize = 11,
		maxsum = wsize*wsize,
	};
	const double thresh = 0.3*maxsum;
	
	src = (float*)_src.data;
	tmp = (float*)_tmp.data;
	
	// run box filter on image containing (-1, 1, 99)
	for(int i = 0; i < (int)_src.total(); i++){
		if(isnan(src[i]))
			tmp[i] = maxsum+1;
		else if(src[i] < 0)
			tmp[i] = -1;
		else
			tmp[i] = 1;
	}
	boxFilter(_tmp, _tmp, -1, Size(wsize,wsize), Point(-1,-1), false);
	if(DEBUG) savenc("apmsum.nc", _tmp);
	
	// create output mask based on box filter output
	_dst = (-thresh <= _tmp) & (_tmp <= thresh);
}

int
main(int argc, char **argv)
{
	Mat sst, cmc, bil, anomaly, lat, lon, m14, m15, m16, medf, stdf, blurf,
		acspo, acspo1, dX, dY, sstmag, delta, omega, albedo, BQ, DQ,
		glabels, glabelsnn, feat, lam1, lam2, apm,
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

SAVENC(acspo);
SAVENC(sst);
SAVENC(cmc);
SAVENC(albedo);

	logprintf("computing sstmag, etc....\n");
	delta = m15 - m16;
	omega = m14 - m15;
	anomaly = sst - cmc;
	gradientmag(sst, dX, dY, sstmag);
	localmax(sstmag, lam2, lam1, 1);
SAVENC(m15);
SAVENC(m16);
SAVENC(delta);
SAVENC(omega);
SAVENC(anomaly);
SAVENC(sstmag);
SAVENC(lam2);
	
	Mat tmp1, tmp2, deltamag;
	gradientmag(delta, tmp1, tmp2, deltamag);
SAVENC(deltamag);

	medianBlur(sst, medf, 5);
	stdfilter(sst-medf, stdf, 7);
	//nanblur(sst, blurf, 7);
	easyclouds = (sst < SST_LOW) | (stdf > STD_THRESH);
		//| (abs(sst - blurf) > EDGE_THRESH);
SAVENC(medf);
SAVENC(stdf);
SAVENC(easyclouds);

	//easyfronts = (sst > SST_LOW) & (sstmag > 0.5)
	//	& (stdf < STD_THRESH) & (lam2 < -0.05);

	bilateral(sst, easyclouds, bil, 3, 200);
SAVENC(bil);

	Mat delbil;
	bilateral(delta, easyclouds, delbil, 0.5, 200);
SAVENC(delbil);
	
	Mat bilanom = sst-bil;
	
	logprintf("running plus/minus...\n");
	plusminus(bilanom, apm);
SAVENC(apm);

	logprintf("quantizing sst delta...\n");
	Var *qinput[] = {new BilAnom(bilanom), new Delta(delta)};
	QVar *qoutput[] = {new QVar(BQ), new QVar(DQ)};
	quantize(nelem(qinput), qinput, omega, sstmag, deltamag, qoutput);
	BQ = qoutput[0]->mat;
	DQ = qoutput[1]->mat;

	logprintf("computing quantized features...\n");
	Var *vars[NFEAT];
	vars[FEAT_LAT] = new Lat(lat);
	vars[FEAT_LON] = new Lon(lon);
	vars[FEAT_SST] = new SST(sst);
	vars[FEAT_DELTA] = new Delta(delta);
	nclust = cluster(qoutput[0], qoutput[1], vars, glabels, feat);
SAVENC(glabels);

	glabelsnn = glabels.clone();
	labelnbrs(feat, vars, easyclouds, sstmag, glabelsnn);
SAVENC(glabelsnn);
	
	logprintf("finding thermal fronts...\n");
	findfronts(lam2, sstmag, stdf, deltamag, glabels, glabelsnn, easyclouds, apm, fronts);

	logprintf("finding clusters adjacent to fronts...\n");
	findadjacent(dY, dX, sstmag, nclust, glabelsnn, acspo, 5, fronts, adjclust);
SAVENC(fronts);

	logprintf("creating spt mask...\n");
	getspt(r, acspo, glabelsnn, adjclust, fronts, spt);
SAVENC(spt);

	logprintf("saving spt mask...\n");
	spt1 = resample_unsort(r->sind, spt);
	writespt(ncid, spt1);
	
	acspo1 = resample_unsort(r->sind, acspo);
	diffcloudmask(acspo1, spt1, diff);
SAVENC(diff);
	cvtColor(diff, diff, CV_RGB2BGR);
	resize(diff, diff, Size(), 1/6.0, 1/6.0, INTER_AREA);
	imwrite("diff.png", diff);

	n = nc_close(ncid);
	if(n != NC_NOERR)
		ncfatal(n, "nc_close failed for %s", path);

	delete r;
	logprintf("done\n");
	return 0;
}

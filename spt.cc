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

/*
 * TODO: relabels the fronts:
 * fronts = fronts * [dilated fronts]
 * Then, remove smalls fronts by connected components, etc.
 */

#include "spt.h"
#include "fastBilateral.hpp"

// features
enum {
	FEAT_LAT,
	FEAT_LON,
	FEAT_SST,
	FEAT_DELTA,
	FEAT_OMEGA,
	NFEAT,
};

// front statistics
typedef struct FrontStat FrontStat;
struct FrontStat {
	int size;	// front size in pixels
	double sstmag;	// average SST gradient magnitude
	
	int lsize, rsize;	// size of left and right sides
	double lsst, rsst;	// average SST of left and right sides
	double ldelta, rdelta;	// average delta of left and right sides
	double lsstanom, rsstanom;	// average SST anomaly
	int lcloud, rcloud;	// number of ACSPO cloud pixels
	
	int ndiff;
	double sstdiffmean;	// mean of SST difference between left and right sides
	double sstdiffvar;	// variance of SST difference between left and right sides

	bool ok;		// do we want this front?
};

typedef struct Front Front;
struct Front {
	vector<int>	ind;
	vector<int>	leftind, rightind;
	bool accept;
};


// thernal fronts and their sides
enum {
	FRONT_INVALID = -1,
	FRONT_INIT = 0,	// initial fronts (stage 1)
	FRONT_BIG,	// big enough fronts (stage 2)
	FRONT_OK,	// final fronts (stage 3)
	FRONT_THIN,	// thinned fronts (stage 4)
	FRONT_LEFT,	// left side
	FRONT_RIGHT,	// right side
};


void
frontstatsmat(vector<FrontStat> &v, Mat &dst)
{
	dst.create(v.size(), 9, CV_32FC1);
	for(int i = 0; i < (int)v.size(); i++){
		dst.at<float>(i, 0) = v[i].lsst;
		dst.at<float>(i, 1) = v[i].rsst;
		dst.at<float>(i, 2) = v[i].ldelta;
		dst.at<float>(i, 3) = v[i].rdelta;
		dst.at<float>(i, 4) = v[i].lsstanom;
		dst.at<float>(i, 5) = v[i].rsstanom;
		dst.at<float>(i, 6) = v[i].lcloud / (double)v[i].lsize;
		dst.at<float>(i, 7) = v[i].rcloud / (double)v[i].rsize;
		dst.at<float>(i, 8) = v[i].sstdiffvar;
	}
}

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
		scalefeat = 1.0;
	};
	int quantize(float val) {
		return cvRound((val - DELTA_LOW) * (1.0/DQ_STEP));
	};
};

class Omega : public Var
{
public:
	Omega(Mat &m) {
		mat = m;
		min = OMEGA_LOW;
		max = OMEGA_HIGH;
		avgfeat = true;
		scalefeat = 1.0;
	};
	int quantize(float val) {
		return cvRound((val - OMEGA_LOW) * (1.0/OQ_STEP));
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

class SSTAnom : public Var
{
public:
	SSTAnom(Mat &m) {
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
quantize(int n, Var **src, const Mat &_easyclouds, const Mat &_deltarange,
	const Mat &_omega, const Mat &_sstmag,
	const Mat &_deltamag, QVar **dst)
{
	int i, k;
	bool ok;
	uchar *easyclouds;
	float *deltarange, *omega, *sstmag, *deltamag;
	
	CV_Assert(n > 0);
	Size size = src[0]->mat.size();
	
	for(k = 0; k < n; k++)
		CHECKMAT(src[k]->mat, CV_32FC1);
	for(k = 0; k < n; k++){
		dst[k]->mat.create(size, CV_16SC1);
		dst[k]->mat = Scalar(-1);
	}
	
	CHECKMAT(_easyclouds, CV_8UC1);
	CHECKMAT(_deltarange, CV_32FC1);
	CHECKMAT(_omega, CV_32FC1);
	CHECKMAT(_sstmag, CV_32FC1);
	CHECKMAT(_deltamag, CV_32FC1);
	easyclouds = _easyclouds.data;
	deltarange = (float*)_deltarange.data;
	omega = (float*)_omega.data;
	sstmag = (float*)_sstmag.data;
	deltamag = (float*)_deltamag.data;
	
	// quantize variables
	for(i = 0; i < (int)size.area(); i++){
		if(easyclouds[i]
		|| deltarange[i] > DELTARANGE_THRESH
		|| sstmag[i] > GRAD_LOW		// || delta[i] < -0.5
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

/*
TODO:
where ACSPO says water: (acspo>>6) == 0,
create 2D histogram of SST vs. Delta
remove clusters for which (average SST, average delta)
fall in a low dense area in the histogram
*/

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
	dilate(_glabels >= 0, _labdil, getStructuringElement(MORPH_RECT, Size(101, 101)));
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
		)//|| (sstmag[i] < GRAD_LOW && glabels[i] != COMP_SPECKLE))
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

// Write spt into NetCDF dataset ncid as variable named "spt_mask".
//
static void
writespt(int ncid, const Mat &spt)
{
	int i, n, varid, ndims, dimids[2];
	nc_type xtype;
	size_t len;
	
	CHECKMAT(spt, CV_8UC1);
	
	const char varname[] = "spt_mask";
	const char varunits[] = "none";
	const char vardescr[] = "SPT mask packed into 1 byte: bits1-2 (00=clear; 01=probably clear; 10=cloudy; 11=clear-sky mask undefined); bit3 (0=no thermal front; 1=thermal front)";

	// chunk sizes used by acspo_mask
	const size_t chunksizes[] = {1024, 3200};
	
	// It's not possible to delete a NetCDF variable, so attempt to use
	// the variable if it already exists. Create the variable if it does not exist.
	n = nc_inq_varid(ncid, varname, &varid);
	if(n != NC_NOERR){
		n = nc_inq_dimid(ncid, "scan_lines_along_track", &dimids[0]);
		if(n != NC_NOERR)
			ncfatal(n, "nc_inq_dimid failed");

		n = nc_inq_dimid(ncid, "pixels_across_track", &dimids[1]);
		if(n != NC_NOERR)
			ncfatal(n, "nc_inq_dimid failed");
		
		n = nc_def_var(ncid, varname, NC_UBYTE, nelem(dimids), dimids, &varid);
		if(n != NC_NOERR)
			ncfatal(n, "nc_def_var failed");
		n = nc_def_var_chunking(ncid, varid, NC_CHUNKED, chunksizes);
		if(n != NC_NOERR)
			ncfatal(n, "nc_def_var_chunking failed");
		n = nc_def_var_deflate(ncid, varid, 0, 1, 1);
		if(n != NC_NOERR)
			ncfatal(n, "setting deflate parameters failed");
		
		n = nc_put_att_text(ncid, varid, "UNITS", nelem(varunits)-1, varunits);
		if(n != NC_NOERR)
			ncfatal(n, "setting attribute UNITS failed");
		n = nc_put_att_text(ncid, varid, "Description", nelem(vardescr)-1, vardescr);
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

nc_type
cv2nctype(int cvtype)
{
	nc_type t;

	switch(cvtype){
	default:
		t = NC_NAT;	// not a type
		break;
	case CV_32FC1:
		t = NC_FLOAT;
		break;
	case CV_8UC1:
		t = NC_UBYTE;
		break;
	}
	return t;
}

static void
createvar(int ncid, const char *varname, const Mat &data)
{
	int i, n, varid, ndims, dimids[2];
	nc_type xtype;
	size_t len;
	
	CV_Assert(data.isContinuous());

	// It's not possible to delete a NetCDF variable, so attempt to use
	// the variable if it already exists. Create the variable if it does not exist.
	n = nc_inq_varid(ncid, varname, &varid);
	if(n != NC_NOERR){
		const char varunits[] = "none";
		const char vardescr[] = "";

		n = nc_inq_dimid(ncid, "scan_lines_along_track", &dimids[0]);
		if(n != NC_NOERR)
			ncfatal(n, "nc_inq_dimid failed");

		n = nc_inq_dimid(ncid, "pixels_across_track", &dimids[1]);
		if(n != NC_NOERR)
			ncfatal(n, "nc_inq_dimid failed");
		
		xtype = cv2nctype(data.type());
		if(xtype == NC_NAT){
			eprintf("unsupported type %s\n", type2str(data.type()));
		}
		n = nc_def_var(ncid, varname, xtype, nelem(dimids), dimids, &varid);
		if(n != NC_NOERR)
			ncfatal(n, "nc_def_var failed");
		n = nc_def_var_deflate(ncid, varid, 0, 1, 1);
		if(n != NC_NOERR)
			ncfatal(n, "setting deflate parameters failed");
		
		n = nc_put_att_text(ncid, varid, "UNITS", nelem(varunits)-1, varunits);
		if(n != NC_NOERR)
			ncfatal(n, "setting attribute UNITS failed");
		n = nc_put_att_text(ncid, varid, "Description", nelem(vardescr)-1, vardescr);
		if(n != NC_NOERR)
			ncfatal(n, "setting attribute Description failed");
	}
	
	// Varify that the netcdf variable has correct type and dimensions.
	n = nc_inq_var(ncid, varid, NULL, &xtype, &ndims, dimids, NULL);
	if(n != NC_NOERR)
		ncfatal(n, "nc_inq_var failed");
	if(cv2nctype(data.type()) != xtype)
		eprintf("invalid variable type %d\n", xtype);
	if(ndims != 2)
		eprintf("variable dims is %d, want 2\n", ndims);
	for(i = 0; i < 2; i++){
		n = nc_inq_dimlen(ncid, dimids[i], &len);
		if(n != NC_NOERR)
			ncfatal(n, "nc_inq_dimlen failed");
		if(len != (size_t)data.size[i])
			eprintf("dimension %d is %d, want %d\n", i, len, data.size[i]);
	}
	
	// Write data into netcdf variable.
	n = nc_put_var(ncid, varid, data.data);
	if(n != NC_NOERR)
		ncfatal(n, "nc_put_var failed");
}

static void
writevar(int ncid, const char *varname, const Mat &data)
{
	int n, varid, ndims, dimids[2];
	nc_type xtype;
	size_t len;
	
	CV_Assert(data.isContinuous());

	n = nc_inq_varid(ncid, varname, &varid);
	if(n != NC_NOERR)
		ncfatal(n, "nc_inq_varid failed");

	// Varify that the netcdf variable has correct type and dimensions.
	n = nc_inq_var(ncid, varid, NULL, &xtype, &ndims, dimids, NULL);
	if(n != NC_NOERR)
		ncfatal(n, "nc_inq_var failed");
	switch(xtype){
	default:
		eprintf("invalid variable type %d\n", xtype);
		break;
	case NC_UBYTE:
		if(data.type() != CV_8UC1)
			eprintf("invalid Mat type %s", type2str(data.type()));
		break;
	case NC_FLOAT:
		if(data.type() != CV_32FC1)
			eprintf("invalid Mat type %s", type2str(data.type()));
		break;
	}
	if(ndims != 2)
		eprintf("variable dims is %d, want 2\n", ndims);
	for(int i = 0; i < 2; i++){
		n = nc_inq_dimlen(ncid, dimids[i], &len);
		if(n != NC_NOERR)
			ncfatal(n, "nc_inq_dimlen failed");
		if(len != (size_t)data.size[i])
			eprintf("dimension %d is %d, want %d\n", i, len, data.size[i]);
	}

	// Write data into netcdf variable.
	n = nc_put_var(ncid, varid, data.data);
	if(n != NC_NOERR)
		ncfatal(n, "nc_put_var failed");
}

// Find thermal fronts.
//
// _lam2 -- local max
// _sstmag -- gradient magnitude
// _stdf -- stdfilter(sst - medianBlur(sst))
// _deltamag -- gradient magnitude of delta
// _glabels -- cluster labels before nearest neighbor lookup
// _glabelsnn -- cluster labels after nearest neighbor lookup
// _easyclouds -- easy clouds
// _anomzero -- zero crossings of anomaly
// _fronts -- thermal fronts (output)
//
static void
findfronts(const Mat &_lam2, const Mat &_sstmag, const Mat &_sstanom, const Mat &_stdf,
	const Mat &_deltamag, const Mat &easyclouds, const Mat &_anomzero, Mat &_fronts)
{
	Mat _dilc, _dilq;
	float *lam2, *sstmag, *stdf, *deltamag;
	double m, llam, lmag, lstdf, ldel;
	int i;
	uchar *dilc, *anomzero;
	schar *fronts;
	
	CHECKMAT(_lam2, CV_32FC1);
	CHECKMAT(_sstmag, CV_32FC1);
	CHECKMAT(_sstanom, CV_32FC1);
	CHECKMAT(_stdf, CV_32FC1);
	CHECKMAT(_deltamag, CV_32FC1);
	CHECKMAT(easyclouds, CV_8UC1);
	CHECKMAT(_anomzero, CV_8UC1);
	_fronts.create(_sstmag.size(), CV_8SC1);
	
	lam2 = (float*)_lam2.data;
	sstmag = (float*)_sstmag.data;
	float *sstanom = (float*)_sstanom.data;
	stdf = (float*)_stdf.data;
	deltamag = (float*)_deltamag.data;
	fronts = (schar*)_fronts.data;
	anomzero = _anomzero.data;
	
	// dilate easyclouds
	dilate(easyclouds, _dilc, getStructuringElement(MORPH_RECT, Size(7, 7)));
	CHECKMAT(_dilc, CV_8UC1);
	dilc = _dilc.data;
/*	
	float *_dilq = 100*(_deltamag - 0.05);
	exp(_dilq, _dilq);
	erode(1.0/(1+_dilq) > 0.5, _dilq, getStructuringElement(MORPH_RECT, Size(7, 7)));
	CHECKMAT(_dilq, CV_32FC1);
	dilq = (float*)_dilq.data;
*/
	// compute thermal fronts image
	for(i = 0; i < (int)_sstmag.total(); i++){
		fronts[i] = FRONT_INVALID;
		
		// continue if in (dilated) easyclouds
		// or not in domain added by nearest neighbor
		if(dilc[i])	// || glabelsnn[i] < 0 || glabels[i] >= 0)
			continue;

		// detect front based on sstmag, deltamag, anomaly
		if(anomzero[i] && sstmag[i] > 0.1 && sstmag[i]/deltamag[i] > 10){
			fronts[i] = FRONT_INIT;
			continue;
		}
		
		// detect front based on sstmag, deltamag, lam2
		m = sstmag[i];
		if(m > 1)
			m = 1;
		ldel = 1.0/(1+exp(100*(deltamag[i]-0.05)));
		llam = 1.0/(1+exp(100*(lam2[i]+0.01)));
		if(m*ldel*llam > 0.5){
			fronts[i] = FRONT_INIT;
			continue;
		}
		
		// detect front based on sstmag, deltamag, lam2, stdf
		lmag = 1.0/(1+exp(-30*(sstmag[i]-0.15)));
		ldel = 1.0/(1+exp(100*(deltamag[i]-0.1)));
		lstdf = 1.0/(1+exp(30*(stdf[i]-0.15)));
		if(lmag*ldel*llam*lstdf > 0.5)
			fronts[i] = FRONT_INIT;
	}
}

// Attempt to connect broken fronts by using cos-similarity of gradient vectors.
// Each pixels within a window is compared to the pixel at the center of the
// window.
// Prototype: matlab/front_connect.m
//
// _fronts -- fronts containing only initial fronts (FRONT_INIT) (intput & output)
// _dX -- gradient in x direction
// _dY -- gradient in y direction
// _sstmag -- SST gradient magnitude
// _easyclouds -- guaranteed cloud based on various thresholds
// lam2 -- local max
//
void
connectfronts(Mat &_fronts, const Mat &_dX, const Mat &_dY,
	const Mat &_sstmag, const Mat &_easyclouds, const Mat &lam2)
{
	CHECKMAT(_fronts, CV_8SC1);
	CHECKMAT(_dX, CV_32FC1);
	CHECKMAT(_dY, CV_32FC1);
	CHECKMAT(_sstmag, CV_32FC1);
	CHECKMAT(_easyclouds, CV_8UC1);
	
	char *fronts = (char*)_fronts.data;
	float *dX = (float*)_dX.data;
	float *dY = (float*)_dY.data;
	
	enum {
		W = 21,	// window width/height
	};
	// pixel at center of window
	const int mid = _fronts.cols*(W/2) + (W/2);
	
	for(int iter = 0; iter < 5; iter++){
		//Mat _valid = (_fronts != FRONT_INIT) & (_sstmag > 0.05)
		//	& (_easyclouds == 0) & (lam2 < LAM2_THRESH);
		Mat _valid = (_fronts != FRONT_INIT) & (_easyclouds == 0);
		CHECKMAT(_valid, CV_8UC1);
		uchar *valid = (uchar*)_valid.data;
		
		int i = 0;
		// For each pixel with full window, where the pixel
		// is top left corner of the window
		for(int y = 0; y < _fronts.rows-W+1; y++){
			for(int x = 0; x < _fronts.cols-W+1; x++){
				if(fronts[i + mid] == FRONT_INIT){
					double cdY = dY[i + mid];
					double cdX = dX[i + mid];
					double max = 0;
					int k = i;
					int argmax = i + mid;
					for(int yy = y; yy < y+W; yy++){
						for(int xx = x; xx < x+W; xx++){
							// cos-similarity
							double sim = dY[k]*cdY + dX[k]*cdX;
							if(valid[k] != 0 && sim > max){
								max = sim;
								argmax = k;
							}
							k++;
						}
						k += _fronts.cols - W;
					}
					fronts[argmax] = FRONT_INIT;
				}
				i++;
			}
		}
	}
}

void
dilatefronts(const Mat &fronts, const Mat &_sstmag, const Mat &_easyclouds, Mat &dst)
{
	Mat _tmp, _bigfronts;
	
	CHECKMAT(fronts, CV_8SC1);
	CHECKMAT(_sstmag, CV_32FC1);
	CHECKMAT(_easyclouds, CV_8UC1);
	_tmp.create(fronts.size(), CV_32FC1);
	
	float *sstmag = (float*)_sstmag.data;
	uchar *easyclouds = _easyclouds.data;
	float *tmp = (float*)_tmp.data;

	connectedComponentsWithLimit(fronts==FRONT_INIT, 8, 9, _bigfronts);
	CHECKMAT(_bigfronts, CV_32SC1);
	int *bigfronts = (int*)_bigfronts.data;
	
	for(int i = 0; i < (int)fronts.total(); i++){
		if(sstmag[i] < 0.1 || easyclouds[i])
			tmp[i] = -1;
		else if(bigfronts[i] >= 0)
			tmp[i] = 1;
		else
			tmp[i] = 0;
	}
	dilate(_tmp, dst, getStructuringElement(MORPH_RECT, Size(21, 21)));

	connectedComponentsWithLimit((dst==1) & (_sstmag > 0.1) & (_easyclouds == 0), 8, 200, dst);
}

// Fill buffer buf with values in values at indices ind
//
void
setvalues(vector<int> &ind, float *buf, float *values)
{
	for(int i = 0; i < (int)ind.size(); i++){
		buf[i] = values[ind[i]];
	}
}

// Set the value at indices ind in buffer buf to value.
///
void
setvalue(vector<int> &ind, char *buf, char value)
{
	for(int i = 0; i < (int)ind.size(); i++){
		buf[ind[i]] = value;
	}
}

// Accept/reject fronts based on front size and front statistics.
//
// _frontsimg -- fronts image with initial front
// _m15 -- band 15 image
// _delta -- band 15 minus band 16
// _dy -- column-wise gradient
// _dx -- row-wise gradient
// _sstmag -- SST gradient magnitude
// _clust -- clustering labels
// _acspo -- ACSPO mask
// _fronts -- fronts information (input & output)
//
static void
verifyfronts(Mat &_frontsimg, const Mat &_m15, const Mat &_delta, const Mat &_omega,
	const Mat &_dy, const Mat &_dx, const Mat &_sstmag, const Mat &_sstanom,
	const Mat &_clust, const Mat &_acspo, vector<Front> &fronts)
{
	Mat _cclabels;

	CHECKMAT(_frontsimg, CV_8SC1);
	CHECKMAT(_m15, CV_32FC1);
	CHECKMAT(_delta, CV_32FC1);
	CHECKMAT(_omega, CV_32FC1);
	CHECKMAT(_dy, CV_32FC1);
	CHECKMAT(_dx, CV_32FC1);
	CHECKMAT(_sstmag, CV_32FC1);
	CHECKMAT(_sstanom, CV_32FC1);
	CHECKMAT(_clust, CV_32SC1);
	CHECKMAT(_acspo, CV_8UC1);
	
	// run connected components on fronts, eliminating small fronts
	int nfront = connectedComponentsWithLimit(_frontsimg==FRONT_INIT, 8, 50, _cclabels);
	if(nfront <= 0)
		return;
	CHECKMAT(_cclabels, CV_32SC1);
	int *cclabels = (int*)_cclabels.data;
	
	fronts.resize(nfront);
	
	// find front indices
	for(int i = 0; i < (int)_frontsimg.total(); i++){
		int lab = cclabels[i];
		if(lab < 0){
			continue;
		}
		fronts[lab].ind.push_back(i);
	}

	float *m15 = (float*)_m15.data;
	float *delta = (float*)_delta.data;
	float *omega = (float*)_omega.data;
	float *dy = (float*)_dy.data;
	float *dx = (float*)_dx.data;
	float *sstmag = (float*)_sstmag.data;
	float *sstanom = (float*)_sstanom.data;
	int *clust = (int*)_clust.data;
	uchar *acspo = (uchar*)_acspo.data;

	// find indices of left and right sides of the fronts
	int i = 0;
	for(int y = 0; y < _frontsimg.rows; y++){
		for(int x = 0; x < _frontsimg.cols; x++){
			int lab = cclabels[i];
			if(lab >= 0){
				// normalize vector (dy, dx) and multiply it by FRONT_SIDE_DIST
				int dy1 = round(FRONT_SIDE_DIST * dy[i]/sstmag[i]);
				int dx1 = round(FRONT_SIDE_DIST * dx[i]/sstmag[i]);
				
				// compute indices of left and right sides
				int left = i + dx1*_frontsimg.cols - dy1;
				int right = i - dx1*_frontsimg.cols + dy1;

				if(0 <= left && left < (int)_frontsimg.total()
				&& (clust[left] >= 0 || (acspo[left]&MaskLand) != 0)
				&& !isnan(m15[left]) && !isnan(delta[left])){
					fronts[lab].leftind.push_back(left);
				}
				if(0 <= right && right < (int)_frontsimg.total()
				&& (clust[right] >= 0 || (acspo[right]&MaskLand) != 0)
				&& !isnan(m15[right]) && !isnan(delta[right])){
					fronts[lab].rightind.push_back(right);
				}
			}
			i++;
		}
	}
	
	Mat _frontm15 = Mat::zeros(_frontsimg.total(), 1, CV_32FC1);
	Mat _frontdelta = Mat::zeros(_frontsimg.total(), 1, CV_32FC1);
	Mat _frontomega = Mat::zeros(_frontsimg.total(), 1, CV_32FC1);
	Mat _frontsstanom = Mat::zeros(_frontsimg.total(), 1, CV_32FC1);
	Mat _leftdelta = Mat::zeros(_frontsimg.total(), 1, CV_32FC1);
	Mat _rightdelta = Mat::zeros(_frontsimg.total(), 1, CV_32FC1);
	float *frontm15 = (float*)_frontm15.data;
	float *frontdelta = (float*)_frontdelta.data;
	float *frontomega = (float*)_frontomega.data;
	float *frontsstanom = (float*)_frontsstanom.data;
	float *leftdelta = (float*)_leftdelta.data;
	float *rightdelta = (float*)_rightdelta.data;
	
	// mark accepted fronts based on front statistics
	for(int lab = 0; lab < nfront; lab++){
		Front &f = fronts[lab];
		f.accept = false;
		
		// TODO: uncomment?
		//setvalues(f.ind, frontsstanom, sstanom);
		//if(maxn(frontsstanom, f.ind.size()) < -0.3){
		//	continue;
		//}
		
		// Reject front if
		//	abs(delta1 + delta2 - 2*delta_f) >= 0.04
		// where delta1 is average delta at one side, delta2 at the other side,
		// and delta_f at the front.
		setvalues(f.ind, frontdelta, delta);
		setvalues(f.leftind, leftdelta, delta);
		setvalues(f.rightind, rightdelta, delta);
		double ddiff = fabs(meann(leftdelta, f.leftind.size())
			+ meann(rightdelta, f.rightind.size())
			-2*meann(frontdelta, f.ind.size()));
		if(ddiff >= 0.04){
			continue;
		}

		// Reject front based on correlation coefficient at the front.
		setvalues(f.ind, frontm15, m15);
		setvalues(f.ind, frontomega, omega);
		if(corrcoef(frontm15, frontdelta, f.ind.size()) <= 0
		|| corrcoef(frontomega, frontdelta, f.ind.size()) >= 0){
			continue;
		}
		
		f.accept = true;
	}
}

// Add left/right sides of fronts in fronts image, and mark accepted fronts.
//
// fronts -- front information
// _frontsimg -- fronts image with initial front (input & ouptut)
//
void
updatefrontsimg(vector<Front> &fronts, const Mat & _frontsimg)
{
	CHECKMAT(_frontsimg, CV_8SC1);
	char *frontsimg = (char*)_frontsimg.data;
	
	for(int i = 0; i < (int)fronts.size(); i++){
		Front &f = fronts[i];
		
		if(f.accept){
			setvalue(f.ind, frontsimg, FRONT_OK);
		}else{
			setvalue(f.ind, frontsimg, FRONT_BIG);
		}
		setvalue(f.leftind, frontsimg, FRONT_LEFT);
		setvalue(f.rightind, frontsimg, FRONT_RIGHT);
	}
}

// Thin fronts based on SST gradient.
// Prototype: matlab/front_thinning.m
//
// frontsimg -- front labels
// dY -- gradient in Y direction
// dX -- gradient in X direction
// sstmag -- gradient magnitude
// thinnedf -- front labels containing thinned fronts (output)
//
void
thinfronts(const Mat &_frontsimg, const Mat &_dY, const Mat &_dX, const Mat &_sstmag, const Mat &_spt, Mat &thinnedf)
{
	CHECKMAT(_frontsimg, CV_8SC1);
	CHECKMAT(_dX, CV_32FC1);
	CHECKMAT(_dY, CV_32FC1);
	CHECKMAT(_sstmag, CV_32FC1);
	CHECKMAT(_spt, CV_8UC1);
	
	char *frontsimg = (char*)_frontsimg.data;
	float *dX = (float*)_dX.data;
	float *dY = (float*)_dY.data;
	float *sstmag = (float*)_sstmag.data;
	uchar *spt = _spt.data;
	
	thinnedf = _frontsimg.clone();
	
	int i = 0;
	for(int y = 0; y < _frontsimg.rows; y++){
		for(int x = 0; x < _frontsimg.cols; x++){
			if((spt[i]&MaskCloud) == MaskCloudClear
			&& (frontsimg[i] == FRONT_OK || frontsimg[i] == FRONT_BIG)){
				double dy = dY[i] / sstmag[i];
				double dx = dX[i] / sstmag[i];

				int maxy = y;
				int maxx = x;
				float maxg = _sstmag.at<float>(maxy, maxx);
				for(int alpha = -5; alpha <= 5; alpha++){
					int yy = round(y + alpha*dx);
					int xx = round(x - alpha*dy);
					
					if(0 <= yy && yy < _frontsimg.rows
					&& 0 <= xx && xx < _frontsimg.cols
					&& _sstmag.at<float>(yy,xx) > maxg){
						maxg = _sstmag.at<float>(yy,xx);
						maxx = xx;
						maxy = yy;
					}
					//if(0 <= yy && yy < _frontsimg.rows
					//&& 0 <= xx && x < _frontsimg.cols){
					//	thinnedf.at<char>(yy, xx) = FRONT_THIN;
					//}
				}
				if(_frontsimg.at<char>(maxy, maxx) == FRONT_OK
				|| _frontsimg.at<char>(maxy, maxx) == FRONT_BIG){
					thinnedf.at<char>(maxy, maxx) = FRONT_THIN;
				}
			}
			i++;
		}
	}
}

// Find adjacent clusters of fronts.
//
// fronts -- front information
// nclust -- number of clusters
// _clust -- clustering labels
// _adjclust -- mask indicated if the a cluster is adjacent to a front (output)
//
void
findadjacent2(vector<Front> &fronts, int nclust, const Mat &_clust, Mat &_adjclust)
{
	CHECKMAT(_clust, CV_32SC1);
	int *clust = (int*)_clust.data;

	int countsize[] = {(int)fronts.size(), nclust};
	SparseMat leftcount(nelem(countsize), countsize, CV_32SC1);
	SparseMat rightcount(nelem(countsize), countsize, CV_32SC1);
	
	for(int i = 0; i < (int)fronts.size(); i++){
		Front &f = fronts[i];
		
		for(int j = 0; j < (int)f.leftind.size(); j++){
			(*(int*)leftcount.ptr(i, clust[f.leftind[j]], true))++;
		}
		for(int j = 0; j < (int)f.rightind.size(); j++){
			(*(int*)rightcount.ptr(i, clust[f.rightind[j]], true))++;
		}
	}
	
	_adjclust = Mat::zeros(nclust, 1, CV_8UC1);
	uchar *adjclust = _adjclust.data;

	for(int i = 0; i < (int)fronts.size(); i++){
		Front &f = fronts[i];
		if(!f.accept){
			continue;
		}
		
		for(int j = 0; j < nclust; j++){
			bool inleft = false;
			bool inright = false;
			int *p;

			p = (int*)leftcount.ptr(i, j, false);
			if(p && *p/(double)f.leftind.size() > 0.3)
				inleft = true;

			p = (int*)rightcount.ptr(i, j, false);
			if(p && *p/(double)f.rightind.size() > 0.3)
				inright = true;
			
			if((inleft && !inright) || (!inleft && inright))
				adjclust[j] = 255;
		}
	}
}


// Narrow down the number of thermal fronts and find clusters that are
// adjacent to those fronts.
//
// _dy -- column-wise gradient
// _dx -- row-wise gradient
// _sstmag -- gradient magnitude
// nclust -- number of clusters
// _clust -- clustering labels
// _acspo -- ACSPO mask
// _fronts -- fronts image (input & output)
// _adjclust -- mask indicated if the a cluster is adjacent to a front (output)
//
static void
findadjacent(const Mat &_sst, const Mat &_dy, const Mat &_dx, const Mat &_sstmag,
	const Mat &_sstanom, const Mat &_delta, const Mat &_m15, const Mat &_m16,
	int nclust, const Mat &_clust, const Mat &_acspo,
	Mat &_fronts, Mat &_adjclust)
{
	Mat _cclabels;
	int *p, nfront, y, x, k, left, right, *cclabels;
	double dy1, dx1;
	uchar *adjclust;
	
	CHECKMAT(_sst, CV_32FC1);
	CHECKMAT(_dy, CV_32FC1);
	CHECKMAT(_dx, CV_32FC1);
	CHECKMAT(_sstmag, CV_32FC1);
	CHECKMAT(_sstanom, CV_32FC1);
	CHECKMAT(_delta, CV_32FC1);
	CHECKMAT(_m15, CV_32FC1);
	CHECKMAT(_m16, CV_32FC1);
	CHECKMAT(_clust, CV_32SC1);
	CHECKMAT(_acspo, CV_8UC1);
	CHECKMAT(_fronts, CV_8SC1);
	float *sst = (float*)_sst.data;
	float *dy = (float*)_dy.data;
	float *dx = (float*)_dx.data;
	float *sstmag = (float*)_sstmag.data;
	//float *sstanom = (float*)_sstanom.data;
	float *delta = (float*)_delta.data;
	float *m15 = (float*)_m15.data;
	float *m16 = (float*)_m16.data;
	int *clust = (int*)_clust.data;
	uchar *acspo = (uchar*)_acspo.data;
	schar *fronts = (schar*)_fronts.data;
	
	// initialize output in case we bail early (e.g. if nfront <= 0)
	_adjclust = Mat::zeros(nclust, 1, CV_8UC1);
	
	Mat _m15diff = Mat::zeros(_fronts.size(), CV_32FC1);
	Mat _m16diff = Mat::zeros(_fronts.size(), CV_32FC1);
	Mat _sstdiff = Mat::zeros(_fronts.size(), CV_32FC1);
	_m15diff.setTo(NAN);
	_m16diff.setTo(NAN);
	_sstdiff.setTo(NAN);
	float *m15diff = (float*)_m15diff.data;
	float *m16diff = (float*)_m16diff.data;
	float *sstdiff = (float*)_sstdiff.data;
	
	// run connected components on fronts, eliminating small fronts
	nfront = connectedComponentsWithLimit(_fronts==FRONT_INIT, 8, 200, _cclabels);
	if(nfront <= 0)
		return;
	CHECKMAT(_cclabels, CV_32SC1);
	cclabels = (int*)_cclabels.data;
	if(DEBUG) savenc("flabels.nc", _cclabels);
	logprintf("findadjacent: initial number of fronts: %d\n", nfront);
	
	int countsize[] = {nfront, nclust};
	SparseMat leftcount(nelem(countsize), countsize, CV_32SC1);
	SparseMat rightcount(nelem(countsize), countsize, CV_32SC1);

	vector<FrontStat> fstats(nfront);
	memset(&fstats[0], 0, sizeof(FrontStat)*fstats.size());
	
	// find left and right sides of the fronts, and their statistics
	k = 0;
	for(y = 0; y < _fronts.rows; y++){
		for(x = 0; x < _fronts.cols; x++){
			if(cclabels[k] < 0){
				k++;
				continue;
			}
			
			// normalize vector (dy, dx) and multiply it by FRONT_SIDE_DIST
			dy1 = round(FRONT_SIDE_DIST * dy[k]/sstmag[k]);
			dx1 = round(FRONT_SIDE_DIST * dx[k]/sstmag[k]);
			
			// compute indices of left and right sides
			left = k + dx1*_fronts.cols - dy1;
			right = k - dx1*_fronts.cols + dy1;
			
			// compute statistics of front
			FrontStat &fs = fstats[cclabels[k]];
			fs.size++;
			fs.sstmag += sstmag[k];
			
			fronts[k] = FRONT_BIG;
			if(0 <= left && left < (int)_fronts.total()
			&& (clust[left] >= 0 || (acspo[left]&MaskLand) != 0)
			//&& !isnan(sstanom[left])
			&& !isnan(sst[left]) && !isnan(delta[left])){
				fs.lsize++;
				fs.lsst += sst[left];
				fs.ldelta += delta[left];
				//fs.lsstanom += sstanom[left];
				fronts[left] = FRONT_LEFT;
				(*(int*)leftcount.ptr(cclabels[k], clust[left], true))++;

				int cm = acspo[left]&MaskCloud;
				if(cm != MaskCloudClear)
					fs.lcloud++;
			}
			if(0 <= right && right < (int)_fronts.total()
			&& (clust[right] >= 0 || (acspo[right]&MaskLand) != 0)
			//&& !isnan(sstanom[right])
			&& !isnan(sst[right]) && !isnan(delta[right])){
				fs.rsize++;
				fs.rsst += sst[right];
				fs.rdelta += delta[right];
				//fs.rsstanom += sstanom[right];
				fronts[right] = FRONT_RIGHT;
				(*(int*)rightcount.ptr(cclabels[k], clust[right], true))++;

				int cm = acspo[right]&MaskCloud;
				if(cm != MaskCloudClear)
					fs.rcloud++;
			}
			if(0 <= left && left < (int)_fronts.total()
			&& 0 <= right && right < (int)_fronts.total()
			&& fronts[left] == FRONT_LEFT && fronts[right] == FRONT_RIGHT){
				m15diff[k] = m15[left] - m15[right];
				m16diff[k] = m16[left] - m16[right];
				sstdiff[k] = sst[left] - sst[right];
				
				// Computer variance online.
				// Donald E. Knuth (1998). The Art of Computer Programming, volume 2:
				// Seminumerical Algorithms, 3rd edn., p. 232. Boston: Addison-Wesley.
				// http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
				double v = sst[right] - sst[left];
				double q = v - fs.sstdiffmean;
				fs.ndiff++;
				fs.sstdiffmean += q/fs.ndiff;
				fs.sstdiffvar += q * (v-fs.sstdiffmean);
			}
			k++;
		}
	}
	
	if(DEBUG) savenc("m15diff.nc", _m15diff);
	if(DEBUG) savenc("m16diff.nc", _m16diff);
	if(DEBUG) savenc("sstdiff.nc", _sstdiff);
	
	logprintf("findadjacent: number of pixels left of fronts: %lu\n", leftcount.nzcount());
	logprintf("findadjacent: number of pixels right of fronts: %lu\n", rightcount.nzcount());
	
	adjclust = _adjclust.data;
	
	// find which clusters are adjacent to a front
	for(int i = 0; i < nfront; i++){
		FrontStat &fs = fstats[i];
		fs.sstmag /= fs.size;
		fs.lsst /= fs.lsize;
		fs.rsst /= fs.rsize;
		fs.ldelta /= fs.lsize;
		fs.rdelta /= fs.rsize;
		//fs.lsstanom /= fs.lsize;
		//fs.rsstanom /= fs.rsize;
		if(fs.ndiff < 2)
			fs.sstdiffvar = 0;
		else
			fs.sstdiffvar /= fs.ndiff - 1;
		
		if(DEBUG && i==5){
			logprintf("front %d, cloud %d/%d, side size %d/%d\n", i, fs.lcloud, fs.rcloud, fs.lsize, fs.rsize);
		}
		fs.ok = fabs((fs.ldelta-fs.rdelta) / (fs.lsst-fs.rsst)) < 0.1
			&& (fs.lcloud < 0.9*fs.lsize || fs.rcloud < 0.9*fs.rsize);
		//fs.ok = fs.lsstanom*fs.rsstanom < 0;
		//double t = 0.7*fs.size;
		//fs.ok = fs.lsize > t && fs.rsize > t;
			//&& fs.sstmag > GRAD_THRESH;
		
		//if(fs.lsst  <= fs.rsst)
		//	fs.ok = fs.ok && fs.ldelta <= fs.rdelta;
		//else
		//	fs.ok = fs.ok && fs.rdelta <= fs.ldelta;
		//fs.ok = fs.ok && fabs(fs.ldelta - fs.rdelta) < 0.1;
		
		if(!fs.ok)
			continue;
		
		for(int j = 0; j < nclust; j++){
			bool inleft = false;
			bool inright = false;

			p = (int*)leftcount.ptr(i, j, false);
			if(p && *p/(double)fs.lsize > 0.3)
				inleft = true;

			p = (int*)rightcount.ptr(i, j, false);
			if(p && *p/(double)fs.rsize > 0.3)
				inright = true;
			
			if((inleft && !inright) || (!inleft && inright))
				adjclust[j] = 255;
		}
	}
	
	// set fronts that are accepted in fronts image
	for(int k = 0; k < (int)_fronts.total(); k++){
		int lab = cclabels[k];
		if(lab >= 0 && fstats[lab].ok)
			fronts[k] = FRONT_OK;
	}

	if(DEBUG){
		Mat fstatsmat;
		frontstatsmat(fstats, fstatsmat);
		savenc("frontstats.nc", fstatsmat);
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
resample_acloud(const Resample &r, const Mat &_acspo, Mat &_acloud)
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
	resample_float32(&r, _acloud, _acloud, false);
}

typedef struct Cluster Cluster;
struct Cluster {
	vector<int> ind;
	int bordersize, bordercloud;
	bool accept;
};

// Verify clusters are acceptable.
//
// _labels -- clustering labels
// start -- ignore labels less than start
// nlabels -- the labels in _labels are in the range [0, nlabels)
// stats -- connected component stats for the clusters
// _fronts -- fronts image
// m15, delta -- m15 and delta images
// clusters -- clustering info that's caller allocated (output)
//
static void
verifyclusters(const Mat &_labels, int start, int nlabels, const Mat &stats,
	const Mat &_fronts, const Mat &_m15, const Mat &_delta,
	vector<Cluster> &clusters)
{
	CHECKMAT(_labels, CV_32SC1);
	CHECKMAT(stats, CV_32SC1);
	CHECKMAT(_fronts, CV_8SC1);
	CHECKMAT(_m15, CV_32FC1);
	CHECKMAT(_delta, CV_32FC1);
	int *labels = (int*)_labels.data;
	char *fronts = (char*)_fronts.data;
	float *m15 = (float*)_m15.data;
	float *delta = (float*)_delta.data;
	
	// count number of pixels in ok fronts & all fronts for each component
	Mat _okfronts = Mat::zeros(nlabels, 1, CV_32SC1);
	Mat _totfronts = Mat::zeros(nlabels, 1, CV_32SC1);
	int *okfronts = (int*)_okfronts.data;
	int *totfronts = (int*)_totfronts.data;
	for(int i = 0; i < (int)_fronts.total(); i++){
		int lab = labels[i];
		if(lab < start){
			continue;
		}
		if(fronts[i] == FRONT_OK){
			okfronts[lab]++;
		}
		if(fronts[i] == FRONT_INIT || fronts[i] == FRONT_BIG || fronts[i] == FRONT_OK){
			totfronts[lab]++;
		}
	}
	
	if(false){
		Mat _frontrat = Mat::zeros(_fronts.size(), CV_32FC1);
		float *frontrat = (float*)_frontrat.data;
		for(int i = 0; i < (int)_fronts.total(); i++){
			frontrat[i] = NAN;
			int lab = labels[i];
			if(lab >= start && stats.at<int>(lab, CC_STAT_AREA) >= 100){
				frontrat[i] = okfronts[lab]/(double)totfronts[lab];
			}
		}
		if(DEBUG) savenc("frontrat.nc", _frontrat);
	}
	
	for(int i = 0; i < (int)_labels.total(); i++){
		int lab = labels[i];
		if(lab >= start){
			clusters[lab].ind.push_back(i);
		}
	}
	
	Mat _m15buf = Mat::zeros(_labels.total(), 1, CV_32FC1);
	Mat _deltabuf = Mat::zeros(_labels.total(), 1, CV_32FC1);
	float *m15buf = (float*)_m15buf.data;
	float *deltabuf = (float*)_deltabuf.data;

	for(int lab = start; lab < nlabels; lab++){
		Cluster &c = clusters[lab];
		c.accept = false;
		if(stats.at<int>(lab, CC_STAT_AREA) < 100){
			continue;
		}
		//if(totfronts[lab] == 0 || okfronts[lab]/(double)totfronts[lab] <= 0.25){
		//	continue;
		//}
		
		setvalues(c.ind, m15buf, m15);
		setvalues(c.ind, deltabuf, delta);
		if(corrcoef(m15buf, deltabuf, c.ind.size()) >= 0){
			c.accept = true;
		}
	}
}

static void
removewithincloud(const Mat &_labels32, int start, int nlabels, const Mat &_acloud,
	vector<Cluster> &clusters)
{
	Mat _labels, _labelsdil;
	
	// dilation function can't handle 32-bit integer Mat, so convery to 16-bit
	CHECKMAT(_labels32, CV_32SC1);
	if(nlabels >= (1<<15)-1){
		eprintf("too many clusters %d\n", nlabels);
	}
	_labels32.convertTo(_labels, CV_16SC1);
	dilate(_labels, _labelsdil, getStructuringElement(MORPH_RECT, Size(3, 3)));

	CHECKMAT(_labels, CV_16SC1);
	CHECKMAT(_labelsdil, CV_16SC1);
	CHECKMAT(_acloud, CV_32FC1);
	short *labels = (short*)_labels.data;
	short *labelsdil = (short*)_labelsdil.data;
	float *acloud = (float*)_acloud.data;

	for(int lab = start; lab < nlabels; lab++){
		Cluster &c = clusters[lab];
		c.bordersize = c.bordercloud = 0;
	}
	for(int i = 0; i < (int)_labels.total(); i++){
		int lab = labelsdil[i];
		// count number of cloud pixels on border of accepted clusters
		if(lab >= start && labels[i] < start && clusters[lab].accept){
			clusters[lab].bordersize++;
			if(!isnan(acloud[i]) && acloud[i] > 0){
				clusters[lab].bordercloud++;
			}
		}
	}
	for(int lab = start; lab < nlabels; lab++){
		Cluster &c = clusters[lab];
		c.accept = c.accept && c.bordersize != 0 && c.bordercloud/(double)c.bordersize < 0.9;
	}
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
getspt(const Resample &r, const Mat &_acspo, const Mat &_clust,
	const Mat &_adjclust, const Mat &_fronts, const Mat &m15, const Mat &delta, int frontvalue, Mat &_spt)
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
		if(fronts[i] == frontvalue)
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

	vector<Cluster> clusters(nlab);
	verifyclusters(_labels, 1, nlab, stats, _fronts, m15, delta, clusters);
	removewithincloud(_labels, 1, nlab, _acloud, clusters);
	
	// Restored some clear-sky in spt mask.
	// We disable small components from being restored.
	// Also check ratio of accepted fronts over all fronts is large.
	for(i = 0; i < (int)_acspo.total(); i++){
		n = labels[i];
		if(n > 0 && (acspo[i]&MaskCloud) == MaskCloudSure
		&& clusters[n].accept)
			spt[i] = MaskCloudClear >> MaskCloudOffset;
	}
}

// Bilateral filter wrapper.
static void
bilateral(const Mat &_src, const Mat &_easyclouds, Mat &_dst, double high, double sigmacolor, double sigmaspace)
{
	int i;
	Mat _tmp;
	float *src, *tmp, *dst;
	uchar *easyclouds;

	CHECKMAT(_src, CV_32FC1);
	CHECKMAT(_easyclouds, CV_8UC1);
	_tmp = _src.clone();
	
	easyclouds = _easyclouds.data;
	src = (float*)_src.data;
	tmp = (float*)_tmp.data;
	
	for(i = 0; i < (int)_src.total(); i++){
		if(easyclouds[i])
			tmp[i] = -1;
		else if(src[i] > high)
			tmp[i] = high;
	}
	// TODO: check if OpenCV's bilateralFilter is fast enough for us
	// and can replace this function.
	cv_extend::bilateralFilter(_tmp, _dst, sigmacolor, sigmaspace);

	CHECKMAT(_dst, CV_32FC1);
	dst = (float*)_dst.data;
	for(i = 0; i < (int)_src.total(); i++){
		if(easyclouds[i])
			dst[i] = NAN;
	}
}

// Find the zero crossings of an image (the edges between negative and positive
// values of source image).
// 
// _src -- source image
// _dst -- destination mask image (output)
//
static void
zerocrossing(const Mat &_src, Mat &_dst)
{
	Mat _tmp;
	float *src, *tmp;
	
	CHECKMAT(_src, CV_32FC1);
	_tmp.create(_src.size(), CV_32FC1);
	
	enum {
		wsize = 11,
		maxsum = +wsize*wsize,
		minsum = -wsize*wsize,
	};
	const double thresh = 0.3*maxsum;
	
	src = (float*)_src.data;
	tmp = (float*)_tmp.data;
	
	// run box filter on image containing {-1, 1, inf}
	for(int i = 0; i < (int)_src.total(); i++){
		if(isnan(src[i])){
			// large enough so that the sum within a window
			// is greater than maxsum
			tmp[i] = -minsum+maxsum+1;
		}else if(src[i] < 0)
			tmp[i] = -1;
		else
			tmp[i] = 1;
	}
	boxFilter(_tmp, _tmp, -1, Size(wsize,wsize), Point(-1,-1), false);
	
	// create output mask based on box filter output
	_dst = (-thresh <= _tmp) & (_tmp <= thresh);
}

void
groweasyclouds(Mat &easyclouds, const Mat &deltarange)
{
	Mat dil;
	
	Mat selem = getStructuringElement(MORPH_RECT, Size(5, 5));
	Mat bigrange = deltarange > DELTARANGE_THRESH;
	
	for(int i = 0; i < 10; i++){
		dilate(easyclouds, dil, selem);
		easyclouds |= dil & bigrange;
	}
}

int
main(int argc, char **argv)
{
	Mat dX, dY, sstmag, deltamag, deltarange, lam1, lam2,
		medf, stdf, blurf,
		sstbil, deltabil, anomzero,
		glabels, glabelsnn, feat, BQ, DQ,
		frontsimg, adjclust, spt, diff;
	int ncid, n, nclust;
	char *path;
	Resample r;

	if(argc != 2)
		eprintf("usage: %s granule\n", argv[0]);
	path = argv[1];
	logprintf("granule: %s\n", path);
	
	logprintf("reading and resampling...\n");
	ncid = open_resampled(path, &r, NC_WRITE);
	Mat sst = readvar_resampled(ncid, &r, "sst_regression");
	Mat cmc = readvar_resampled(ncid, &r, "sst_reynolds");
	Mat lat = readvar_resampled(ncid, &r, "latitude");
	Mat lon = readvar_resampled(ncid, &r, "longitude");
	Mat acspo = readvar_resampled(ncid, &r, "acspo_mask");
	Mat m14 = readvar_resampled(ncid, &r, "brightness_temp_chM14");
	Mat m15 = readvar_resampled(ncid, &r, "brightness_temp_chM15");
	Mat m16 = readvar_resampled(ncid, &r, "brightness_temp_chM16");
	Mat albedo = readvar_resampled(ncid, &r, "albedo_chM7");
SAVENC(acspo);
SAVENC(sst);
SAVENC(cmc);
SAVENC(m14);
SAVENC(m15);
SAVENC(albedo);

	logprintf("computing sstmag, etc....\n");
	gradientmag(sst, sstmag, dX, dY);
if(DEBUG) savenc("oldsstmag.nc", sstmag);

	logprintf("Laplacian of Gaussican...\n");
	nanlogfilter(sstmag, 17, 2, -17, sstmag);
	sstmag.setTo(0, sstmag < 0);
SAVENC(sstmag);

	Mat delta = m15 - m16;
	Mat omega = m14 - m15;
	gradientmag(omega, deltamag);
	rangefilter(delta, deltarange, 7);
	localmax(sstmag, lam2, lam1, 1);
SAVENC(m15);
SAVENC(m16);
SAVENC(delta);
SAVENC(omega);
SAVENC(deltamag);
SAVENC(deltarange);
SAVENC(lam2);

	logprintf("computing easyclouds...\n");
	medianBlur(sst, medf, 5);
	stdfilter(sst-medf, stdf, 7);
	//nanblur(sst, blurf, 7);
	Mat easyclouds = (sst < SST_LOW) | (stdf > STD_THRESH);
		//| (deltarange > 0.5);
		//| (abs(sst - blurf) > EDGE_THRESH);
	groweasyclouds(easyclouds, deltarange);
	//Mat easyfronts = (sst > SST_LOW) & (sstmag > 0.5)
	//	& (stdf < STD_THRESH) & (lam2 < LAM2_THRESH);
SAVENC(medf);
SAVENC(stdf);
SAVENC(easyclouds);
	
	Mat sstbil1;
	bilateral(sst, easyclouds, sstbil1, SST_HIGH, 3, 21);
	Mat sstanom1 = sst - sstbil1;
SAVENC(sstbil1);
	
	logprintf("computing anomaly zero crossings...\n");
	bilateral(sst, easyclouds, sstbil, SST_HIGH, 3, 200);
	Mat sstanom = sst - sstbil;
	zerocrossing(sstanom1, anomzero);
SAVENC(sstbil);
SAVENC(anomzero);

	logprintf("quantizing variables...\n");
	Var *qinput[] = {new SSTAnom(sstanom), new Delta(delta)};
	QVar *qoutput[] = {new QVar(BQ), new QVar(DQ)};
	quantize(nelem(qinput), qinput, easyclouds, deltarange, omega, sstmag, deltamag, qoutput);
	BQ = qoutput[0]->mat;
	DQ = qoutput[1]->mat;

	logprintf("clustering...\n");
	Var *vars[NFEAT] = {
		new Lat(lat),
		new Lon(lon),
		new SST(sst),
		new Delta(delta),
		new Omega(omega),
	};
	nclust = cluster(qoutput[0], qoutput[1], vars, glabels, feat);
SAVENC(glabels);

	logprintf("labeling neighbors...\n");
	glabelsnn = glabels.clone();
	labelnbrs(feat, vars, easyclouds, sstmag, glabelsnn);
SAVENC(glabelsnn);
	
	logprintf("finding thermal fronts...\n");
	findfronts(lam2, sstmag, sstanom, stdf, deltamag, easyclouds, anomzero, frontsimg);
	if(DEBUG) savenc("oldfronts.nc", frontsimg);
	connectfronts(frontsimg, dX, dY, sstmag, easyclouds, lam2);
	if(DEBUG) savenc("connfronts.nc", frontsimg);
	if(false){	
		Mat dilfronts;
		dilatefronts(frontsimg, sstmag, easyclouds, dilfronts);
		SAVENC(dilfronts);
	}
	
	logprintf("finding clusters adjacent to fronts...\n");
	if(true){
		vector<Front> fronts;
		Mat thinnedf;
		
		verifyfronts(frontsimg, m15, delta, omega, dY, dX, sstmag, sstanom, glabelsnn, acspo, fronts);
		updatefrontsimg(fronts, frontsimg);
		findadjacent2(fronts, nclust, glabelsnn, adjclust);
	}else{
		findadjacent(sst, dY, dX, sstmag, sstanom1, delta, m15, m16, nclust, glabelsnn, acspo, frontsimg, adjclust);
	}
SAVENC(frontsimg);
SAVENC(adjclust);

	logprintf("creating spt mask...\n");
	getspt(r, acspo, glabelsnn, adjclust, frontsimg, m15, delta, FRONT_OK, spt);
SAVENC(spt);

	// TODO: this is temporary
	//Mat thinnedf;
	//thinfronts(frontsimg, dY, dX, sstmag, spt, thinnedf);
	//getspt(r, acspo, glabelsnn, adjclust, thinnedf, m15, delta, FRONT_THIN, spt);

	logprintf("saving spt mask...\n");
	if(true){
		Mat spt1 = resample_unsort(r.sind, spt);
		writespt(ncid, spt1);
		Mat acspo1 = resample_unsort(r.sind, acspo);
		diffcloudmask(acspo1, spt1, diff);
	}else{
		writespt(ncid, spt);
		writevar(ncid, "sst_regression", sst);
		writevar(ncid, "acspo_mask", acspo);
		diffcloudmask(acspo, spt, diff);
	}
SAVENC(diff);

	if(false){
		// TODO: this is temporary
		Mat anom = sst - cmc;
		Mat anom1 = resample_unsort(r.sind, anom);
		createvar(ncid, "sst_anomaly", anom1);
	}

	if(false){
		logprintf("saving diff image as diff.png\n");
		cvtColor(diff, diff, CV_RGB2BGR);
		resize(diff, diff, Size(), 1/6.0, 1/6.0, INTER_AREA);
		imwrite("diff.png", diff);
	}

	n = nc_close(ncid);
	if(n != NC_NOERR)
		ncfatal(n, "nc_close failed for %s", path);
	logprintf("done\n");
	return 0;
}

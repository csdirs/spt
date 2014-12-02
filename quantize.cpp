#include "spt.h"

int
quantize_lat(float lat)
{
	float la;
	
	la = abs(lat);
	if(la < 30)
		return 0;
	if(la < 45)
		return 1;
	if(la < 60)
		return 2;
	return 3;
}

int
quantize_sst(float sst)
{
	return cvRound((sst - SST_LOW) * (1.0/TQ_STEP));
}

int
quantize_delta(float delta)
{
	return cvRound((delta - DELTA_LOW) * (1.0/DQ_STEP));
}

int
quantize_omega(float omega)
{
	return cvRound((omega - OMEGA_LOW) * (1.0/OQ_STEP));
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
	float *lat, *sst, *delta, *omega, *gm, *albedo;
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
		quantize_sst(SST_HIGH) + 1,
		quantize_delta(DELTA_HIGH) + 1,
		quantize_omega(OMEGA_HIGH) + 1,
	};
	cloud.create(4, lutsizes, CV_32SC1);
	cloud = Scalar(0);
	ocean.create(4, lutsizes, CV_32SC1);
	ocean = Scalar(0);
	lut.create(4, lutsizes, CV_8SC1);
	lut = Scalar(LUT_UNKNOWN);
	
	logprintf("LUT size is %dx%dx%dx%d\n", lut.size[0], lut.size[1], lut.size[2], lut.size[3]);
	
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
			tq[i] = quantize_sst(sst[i]);
			dq[i] = quantize_delta(delta[i]);
			oq[i] = quantize_omega(omega[i]);
			li = quantize_lat(lat[i]);
			
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

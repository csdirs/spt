#!/bin/sh

make -j || exit 1

#for f in /cephfs/fhs/data/in/acspo/www.star.nesdis.noaa.gov/pub/sod/sst/micros_data/acspo_nc/npp/2014-07-11/*.nc
#do
#	./spt $f
#done


# "/cephfs/fhs/data/in/acspo/www.star.nesdis.noaa.gov/pub/sod/sst/micros_data/acspo_nc/npp/2014-07-10/ACSPO_V2.30_NPP_VIIRS_2014-07-10_1230-1240_20140713.061812.nc"
# "/cephfs/fhs/data/in/acspo/www.star.nesdis.noaa.gov/pub/sod/osb/ykihai/VIIRS_Samples_for_Irina/Select/ACSPO_V2.30_NPP_VIIRS_2014-06-20_1710-1719_20140623.071032.nc"


#./spt /cephfs/fhs/data/in/acspo/www.star.nesdis.noaa.gov/pub/sod/sst/micros_data/acspo_nc/npp/2014-07-11/ACSPO_V2.30_NPP_VIIRS_2014-07-11_0000-0010_20140714.005638.nc

#DATA_PATH=/home/fhs/data/in/acspo/www.star.nesdis.noaa.gov/pub/sod/osb/ykihai/VIIRS_Samples_for_Irina/Select/ACSPO_V2.30_NPP_VIIRS_2014-06-20_1710-1719_20140623.071032.nc

DATA_DIR=/home/fhs/data/in/acspo/www.star.nesdis.noaa.gov/pub/sod/osb/petrenko/VIIRS/VIIRS/day
#DATA_PATH=$DATA_DIR/ACSPO_V2.31b02_NPP_VIIRS_2014-11-04_1410-1419_20141111.212954.nc
#DATA_PATH=$DATA_DIR/ACSPO_V2.31b02_NPP_VIIRS_2014-11-04_1740-1750_20141111.225405.nc
DATA_PATH=$DATA_DIR/ACSPO_V2.31b02_NPP_VIIRS_2014-10-30_0400-0409_20141111.005748.nc

#for f in $DATA_DIR/*.nc; do
#	dirname=$(echo $(basename $f) | sed 's/ACSPO_.*_NPP_VIIRS_\(.*\)-.*\.nc$/\1_output/')
#	mkdir -p $dirname
#	(cd $dirname && 
#		cp matplotlib_wrap.py loaddata.py &&
#		OMP_NUM_THREADS=24 ../spt $f | tee -a loaddata.py
#	)
#done

cp matplotlib_wrap.py loaddata.py
OMP_NUM_THREADS=24 ./spt $DATA_PATH | tee -a loaddata.py
#OMP_NUM_THREADS=24 ./lut $DATA_DIR/*.nc | tee -a loaddata.py

#for f in ACSPO_V2.31b02_NPP_VIIRS_2014-10-30_0020-0030_20141110.232954.nc \
#	ACSPO_V2.31b02_NPP_VIIRS_2014-10-30_0400-0409_20141111.005748.nc \
#	ACSPO_V2.31b02_NPP_VIIRS_2014-10-30_1910-1919_20141111.065956.nc \
#	ACSPO_V2.31b02_NPP_VIIRS_2014-11-04_0040-0049_20141111.163507.nc \
#	ACSPO_V2.31b02_NPP_VIIRS_2014-11-04_0410-0419_20141111.174756.nc \
#	ACSPO_V2.31b02_NPP_VIIRS_2014-11-04_0710-0719_20141111.185006.nc \
#	ACSPO_V2.31b02_NPP_VIIRS_2014-11-04_1030-1039_20141111.200521.nc \
#	ACSPO_V2.31b02_NPP_VIIRS_2014-11-04_1410-1419_20141111.212954.nc \
#	ACSPO_V2.31b02_NPP_VIIRS_2014-11-04_1710-1719_20141111.224328.nc \
#	ACSPO_V2.31b02_NPP_VIIRS_2014-11-04_1740-1750_20141111.225405.nc; do
#	
#	OMP_NUM_THREADS=24 ./spt $DATA_DIR/$f
#done

#!/bin/sh

make -j || exit 1
export OMP_NUM_THREADS=24

# original data: /cephfs/fhs/data/in/acspo/www.star.nesdis.noaa.gov/pub/sod/osb/petrenko/VIIRS/VIIRS/day
DIR=/cephfs/fhs/data/out/spt/20141226/day
GPATH=$DIR/SPT_ACSPO_V2.31b02_NPP_VIIRS_2014-10-30_0400-0409_20141111.005748.nc

cp matplotlib_wrap.py loaddata.py
#./lut $DIR/*.nc | tee -a loaddata.py
./spt $GPATH | tee -a loaddata.py

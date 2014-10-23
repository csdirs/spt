#!/bin/sh

make || exit 1

#for f in /cephfs/fhs/data/in/acspo/www.star.nesdis.noaa.gov/pub/sod/sst/micros_data/acspo_nc/npp/2014-07-11/*.nc
#do
#	./spt $f
#done


# "/cephfs/fhs/data/in/acspo/www.star.nesdis.noaa.gov/pub/sod/sst/micros_data/acspo_nc/npp/2014-07-10/ACSPO_V2.30_NPP_VIIRS_2014-07-10_1230-1240_20140713.061812.nc"
# "/cephfs/fhs/data/in/acspo/www.star.nesdis.noaa.gov/pub/sod/osb/ykihai/VIIRS_Samples_for_Irina/Select/ACSPO_V2.30_NPP_VIIRS_2014-06-20_1710-1719_20140623.071032.nc"

cat > loaddata.py <<EOF
#!/usr/bin/env python2

import numpy as np

EOF
#./spt /cephfs/fhs/data/in/acspo/www.star.nesdis.noaa.gov/pub/sod/sst/micros_data/acspo_nc/npp/2014-07-11/ACSPO_V2.30_NPP_VIIRS_2014-07-11_0000-0010_20140714.005638.nc
OMP_NUM_THREADS=24 ./spt /home/fhs/data/in/acspo/www.star.nesdis.noaa.gov/pub/sod/osb/ykihai/VIIRS_Samples_for_Irina/Select/ACSPO_V2.30_NPP_VIIRS_2014-06-20_1710-1719_20140623.071032.nc | tee -a loaddata.py

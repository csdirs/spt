TARG = spt
CXX = g++
CXXFLAGS = -g -Wall -Wfatal-errors -march=native -O2 -fopenmp \
	-Ibilateral -Ibilateral/include
LD = g++
LDFLAGS = -fopenmp\
	-lnetcdf\
	-lopencv_core\
	-lopencv_imgproc\
	-lopencv_flann\
	-lopencv_highgui\
	-lfftw3\

OFILES = \
	utils.o\
	io.o\
	resample.o\
	connectedcomponents.o\
	filters.o\
	spt.o\
	bilateral/fft_3D/fft_3D.o\

HFILES = spt.h\
	connectedcomponents.h\
	fastBilateral.hpp\


all: $(TARG)

$(TARG): $(OFILES)
	$(LD) -o $(TARG) $(OFILES) $(LDFLAGS)

%.o: %.cc $(HFILES)
	$(CXX) -c $(CXXFLAGS) $<

bilateral/fft_3D/fft_3D.o:
	cd bilateral/fft_3D && $(MAKE)

clean:
	rm -f $(TARG) $(OFILES)
	cd bilateral && $(MAKE) clean

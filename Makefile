TARG = spt
CXX = g++
CXXFLAGS = -g -Wall -Wfatal-errors -march=native -O2 -fopenmp
LD = g++
LDFLAGS = -fopenmp\
	-lnetcdf\
	-lopencv_core\
	-lopencv_imgproc\
	-lopencv_flann\
	-lopencv_highgui\

OFILES = \
	utils.o\
	io.o\
	resample.o\
	connectedcomponents.o\
	filters.o\
	spt.o\

HFILES = spt.h\
	connectedcomponents.h\
	fastBilateral.hpp\


all: $(TARG)

$(TARG): $(OFILES)
	$(LD) -o $(TARG) $(OFILES) $(LDFLAGS)

%.o: %.cc $(HFILES)
	$(CXX) -c $(CXXFLAGS) $<

clean:
	rm -f $(TARG) $(OFILES)

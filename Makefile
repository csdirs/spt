SPT = spt
CMDIFF = cmdiff
CXX = g++
CXXFLAGS = -g -Wall -Wfatal-errors -march=native -O2 -fopenmp
LD = g++
LDFLAGS_SPT =\
	-fopenmp\
	-lnetcdf\
	-lopencv_core\
	-lopencv_imgproc\
	-lopencv_flann\
	-lopencv_highgui\

LDFLAGS_CMDIFF =\
	-lnetcdf\
	-lopencv_core\
	-lopencv_imgproc\
	-lopencv_highgui\

OFILES_SPT = \
	utils.o\
	io.o\
	resample.o\
	connectedcomponents.o\
	filters.o\
	spt.o\

OFILES_CMDIFF =\
	utils.o\
	io.o\
	resample.o\
	cmdiff.o\

HFILES =\
	spt.h\
	connectedcomponents.h\
	fastBilateral.hpp\


all: $(SPT) $(CMDIFF)

$(SPT): $(OFILES_SPT)
	$(LD) -o $(SPT) $(OFILES_SPT) $(LDFLAGS_SPT)

$(CMDIFF): $(OFILES_CMDIFF)
	$(LD) -o $(CMDIFF) $(OFILES_CMDIFF) $(LDFLAGS_CMDIFF)

%.o: %.cc $(HFILES)
	$(CXX) -c $(CXXFLAGS) $<

clean:
	rm -f $(TARG) $(OFILES_SPT) $(OFILES_CMDIFF)

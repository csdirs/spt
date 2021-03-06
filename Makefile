SPT = spt
CMDIFF = cmdiff
RESAM = resam
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

LDFLAGS_RESAM =\
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

OFILES_RESAM = \
	utils.o\
	io.o\
	resample.o\
	resam.o\

HFILES =\
	spt.h\
	connectedcomponents.h\
	fastBilateral.hpp\


all: $(SPT) $(CMDIFF) $(RESAM)

$(SPT): $(OFILES_SPT)
	$(LD) -o $(SPT) $(OFILES_SPT) $(LDFLAGS_SPT)

$(CMDIFF): $(OFILES_CMDIFF)
	$(LD) -o $(CMDIFF) $(OFILES_CMDIFF) $(LDFLAGS_CMDIFF)

$(RESAM): $(OFILES_RESAM)
	$(LD) -o $(RESAM) $(OFILES_RESAM) $(LDFLAGS_RESAM)

%.o: %.cc $(HFILES)
	$(CXX) -c $(CXXFLAGS) $<

clean:
	rm -f $(SPT) $(CMDIFF) $(RESAM) \
		$(OFILES_SPT) $(OFILES_CMDIFF) $(OFILES_RESAM)

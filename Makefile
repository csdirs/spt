SPT = spt
SAVERESAMPLED = saveresampled
LUT = lut
CXX = g++
CXXFLAGS = -g -Wall -Wfatal-errors -march=native -O2 -fopenmp
LD = g++ -fopenmp
LDFLAGS = -lnetcdf\
	-lopencv_core\
	-lopencv_imgproc\
	-lopencv_highgui\
	-lopencv_contrib\
	-lopencv_flann

SPT_OFILES = \
	utils.o\
	io.o\
	npy.o\
	resample.o\
	connectedcomponents.o\
	filters.o\
	quantize.o\
	spt.o\

SAVEANOMALY_OFILES = \
	utils.o\
	io.o\
	resample.o\
	saveresampled.o\

LUT_OFILES = \
	utils.o\
	io.o\
	npy.o\
	resample.o\
	filters.o\
	quantize.o\
	lut.o\

HFILES = spt.h\
	connectedcomponents.h\
	

all: $(SPT) $(SAVERESAMPLED) $(LUT)

$(SAVERESAMPLED): $(SAVEANOMALY_OFILES)
	$(LD) -o $(SAVERESAMPLED) $(SAVEANOMALY_OFILES) $(LDFLAGS)

$(SPT): $(SPT_OFILES)
	$(LD) -o $(SPT) $(SPT_OFILES) $(LDFLAGS)

$(LUT): $(LUT_OFILES)
	$(LD) -o $(LUT) $(LUT_OFILES) $(LDFLAGS)

%.o: %.cc $(HFILES)
	$(CXX) -c $(CXXFLAGS) $<

clean:
	rm -f $(SPT) $(SAVERESAMPLED) $(SAVEANOMALY_OFILES) $(SPT_OFILES)

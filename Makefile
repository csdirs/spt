SPT = spt
SAVERESAMPLED = saveresampled
CXX = g++
CXXFLAGS = -g -Wall -march=native -O2 -fopenmp
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
	resample.o\
	spt.o\
	connectedcomponents.o\

SAVEANOMALY_OFILES = \
	utils.o\
	io.o\
	resample.o\
	saveresampled.o\

HFILES = spt.h\
	connectedcomponents.h\
	

all: $(SPT) $(SAVERESAMPLED)

$(SAVERESAMPLED): $(SAVEANOMALY_OFILES)
	$(LD) -o $(SAVERESAMPLED) $(SAVEANOMALY_OFILES) $(LDFLAGS)

$(SPT): $(SPT_OFILES)
	$(LD) -o $(SPT) $(SPT_OFILES) $(LDFLAGS)

%.o: %.cpp $(HFILES)
	$(CXX) -c $(CXXFLAGS) $<

clean:
	rm -f $(SPT) $(SAVERESAMPLED) $(SAVEANOMALY_OFILES) $(SPT_OFILES)

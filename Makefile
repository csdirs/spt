SPT = spt
SAVEANOMALY = saveanomaly
CXX = g++
CXXFLAGS = -g -Wall -march=native -O2
LD = g++
LDFLAGS = -lnetcdf\
	-lopencv_core\
	-lopencv_imgproc\
	-lopencv_highgui\
	-lopencv_contrib

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
	saveanomaly.o\

HFILES = spt.h\
	connectedcomponents.h\
	

all: $(SPT) $(SAVEANOMALY)

$(SAVEANOMALY): $(SAVEANOMALY_OFILES)
	$(LD) -o $(SAVEANOMALY) $(SAVEANOMALY_OFILES) $(LDFLAGS)

$(SPT): $(SPT_OFILES)
	$(LD) -o $(SPT) $(SPT_OFILES) $(LDFLAGS)

%.o: %.cpp $(HFILES)
	$(CXX) -c $(CXXFLAGS) $<

clean:
	rm -f $(SPT) $(SAVEANOMALY) $(SAVEANOMALY_OFILES) $(SPT_OFILES)

PROG = spt
CXX = g++
CXXFLAGS = -g -Wall -Wfatal-errors -march=native -O2 -fopenmp
LD = g++
LDFLAGS = -fopenmp\
	-lnetcdf\
	-lopencv_core\
	-lopencv_imgproc\
	-lopencv_flann

OFILES = \
	utils.o\
	io.o\
	resample.o\
	connectedcomponents.o\
	filters.o\
	quantize.o\
	spt.o\

HFILES = spt.h\
	connectedcomponents.h\
	

all: $(PROG)

$(PROG): $(OFILES)
	$(LD) -o $(PROG) $(OFILES) $(LDFLAGS)

%.o: %.cc $(HFILES)
	$(CXX) -c $(CXXFLAGS) $<

clean:
	rm -f $(PROG) $(OFILES)

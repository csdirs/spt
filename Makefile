PROG = spt
CXX = g++
CXXFLAGS = -g -Wall
LD = g++
LDFLAGS = -lnetcdf\
	-lopencv_core\
	-lopencv_imgproc\
	-lopencv_highgui\
	-lopencv_contrib

OFILES = \
	utils.o\
	io.o\
	resample.o\
	spt.o\

HFILES = spt.h

$(PROG): $(OFILES)
	$(LD) -o $(PROG) $(OFILES) $(LDFLAGS)

%.o: %.cpp $(HFILES)
	$(CXX) -c $(CXXFLAGS) $<

clean:
	rm -f $(PROG) $(OFILES)

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
	resample.o\
	spt.o\
	utils.o\

HFILES = spt.h

$(PROG): $(OFILES)
	$(LD) -o $(PROG) $(OFILES) $(LDFLAGS)

%.o: %.c $(HFILES)
	$(CXX) -c $(CXXFLAGS) $<

clean:
	rm -f $(PROG) $(OFILES)
#include "spt.h"

void
ncfatal(int n)
{
	cerr << "Error: " << nc_strerror(n) << endl;
	exit(2);
}

void
fatal(string msg)
{
	cerr << msg << endl;
	exit(2);
}

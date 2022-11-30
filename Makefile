FC=ifort
fparallel=-fopenmp
fsimd=-axCORE-AVX2 -xSSE4.2
FFLAGS= -O2 $(fsimd) $(parallel) -shared -fPIC
OBJ=fsub.so

.SUFFIXES:
main: $(OBJ)
	python setup.py build_ext --inplace
%.so: %.f90
	$(FC) $(FFLAGS) -o $(OBJ) $<
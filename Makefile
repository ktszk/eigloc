# Compiler
FC = ifx

# Compiler flags
fparallel = -fopenmp
# -xHOST: auto-detect host CPU and generate maximally optimized code.
#   Use this when compiling and running on the same machine.
#
# For cluster/HPC use (compile on login node, run on compute nodes), comment
# out -xHOST and uncomment the explicit line instead:
#   fsimd = -xCORE-AVX2 -axCORE-AVX512
#     -xCORE-AVX2:     require AVX2 as baseline (Intel Haswell 2013+, all modern CPUs)
#     -axCORE-AVX512:  generate an additional AVX-512 code path (Skylake-X/Ice Lake+)
fsimd     = -xHOST
FFLAGS    = -O2 $(fsimd) $(fparallel) -shared -fPIC

# Output target
OBJ = fsub.so

.PHONY: all clean
.SUFFIXES:

# Build Fortran shared library, then compile Cython extension in-place
all: $(OBJ)
	python setup.py build_ext --inplace

# Compile %.f90 -> %.so
%.so: %.f90
	$(FC) $(FFLAGS) -o $@ $<

clean:
	rm -f $(OBJ) get_ham.cpython-*.so get_ham.c
	rm -rf build/

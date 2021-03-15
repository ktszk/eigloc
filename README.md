# eigloc
This code calculates excitation of n body localized f orbitals using exact diagonalization.

# Rquirements
In this code, we use following pakages. Please install these using pip etc.

- Cython
- numpy
- scipy
- sympy
- matplotlib

#How to make
In this code, get_ham.pyx is written in Cython.
Hence, We need to complie using the following command.

`python setup.py build_ext --inplace`
Linear Regression
=================

An implementation of vanilla linear regression: it solves the normal equation to determine 
the optimal coefficients.

You can run the example using
```sh
cargo run --features=<BLAS backend>
```
where `<BLAS backend>` has to be either `openblas`, `netlib` or `intel-mkl`. 

If you want to use OpenBLAS:
```sh
cargo run --features=openblas
```

See the following section for more details.

BLAS/LAPACK Backend
===================

This example uses `ndarray-linalg`: it thus requires a BLAS/LAPACK backend to be compiled and executed.

Three BLAS/LAPACK implementations are supported:

- [OpenBLAS](https://github.com/cmr/openblas-src)
  - requires `gfortran` (or another Fortran compiler)
- [Netlib](https://github.com/cmr/netlib-src)
  - requires `cmake` and `gfortran`
- [Intel MKL](https://github.com/termoshtt/rust-intel-mkl) (non-free license, see the linked page)

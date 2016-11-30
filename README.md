# hamsi-v0.1
HAMSI (Hessian Approximated Multiple Subsets Iteration) is an incremental optimization algorithm for large-scale problems. It is a parallelized algorithm, with careful load balancing for better performance. The algorithm uses quadratic approximations and a quasi-Newton method (L-BFGS) to find the optimum in each incremental step.

The code given here is developed for research purposes. The related research article (also in this repository) can be consulted for theoretical and algorithmic details.

The code in this repository is designed for _matrix factorization_: Given an m-by-n sparse matrix M, find two matrices X (m-by-k) and Y (k-by-n) such that their product XY is approximately equal to M. We strive to minimize the root-mean-square error: sqrt(sum (M(i,j) - (XY)(i,j))^2)

## Input data format
HAMSI admits a sparse matrix, represented as follows in a plain text file:
```
ndim
size1 size2
nonzerocount
i j M(i,j)
...
```
where
- `ndim` gives the number of dimensions in the data (always 2 for a matrix);
- `size1` and `size2` give the maximum index in each dimension (aka cardinality), or, number of rows and number of columns;
- `nonzerocount` gives the number of nonzero entries in the matrix;
- `i j M(i,j)` give the index 1, index 2, and the value of the matrix element at that location, repeated `nonzerocount` times.

Example: 1M.dat (MovieLens data with 1 million nonzero entries)
```
2
3883 6040 
1000209
1 1 5
48 1 5
149 1 5
258 1 4
524 1 5
528 1 4
...
```
## Compiling
The code is written in C++. The OpenMP library and GSL (GNU SCientific Library) development files are required.

To compile on the command line:
`g++ -std=c++11 hamsi_sharedmem.cpp -lgsl -lgslcblas -fopenmp -O3 -o hamsi`

## Usage
The `hamsi` executable takes the following command line arguments.
```
hamsi <data_file> <# of threads> <latent dimension> <max # of iters> <max time> <seed>
```

|Argument|Description|
|--------|-----------|
|data_file|The text file containing the sparse matrix in the form described above.|
|# of threads|Number of processor threads should be used in the parallel computation.|
|latent dimension|The product matrices have sizes m-by-k and k-by-n, respectively.|
|max # of iters|Stop the computation after so many iterations [INNER? OUTER?]|
|max time|Stop the computation after so many seconds of wallclock time.|
|seed|The seed for random number generation.|

Example:
`./hamsi ./data/1M.dat 4 50 5000 10 123456`

Note: In the first use of the data file, HAMSI generates a binary copy for more efficient computation and storage. It is stored in the same directory. If you do not erase it, the binary copy will be reused in the next run of the program.

## Output
HAMSI displays the details of iteration (such as time, iteration number and error) at each step on standard output (the screen). The resulting factor matrices are stored in files named `factor1.dat` and `factor2.dat` in tabular text form. File `factor1.dat` has `m` rows and `k` columns, and file `factor2.dat` has `k` rows and `n` columns. Columns are separated with space, rows are separated with newline character.

# hamsi-mf: HAMSI for matrix factorization
HAMSI (Hessian Approximated Multiple Subsets Iteration) is an incremental optimization algorithm for large-scale problems. It is a parallelized algorithm, with careful load balancing for better performance. The algorithm uses quadratic approximations and a quasi-Newton method (L-BFGS) to find the optimum in each incremental step.

The code given here is developed for research purposes. The related research article (ArXiv link to be added) can be consulted for theoretical and algorithmic details.

The code in this repository is designed for **matrix factorization**: Given an m-by-n sparse matrix M, find two matrices X (m-by-k) and Y (k-by-n) such that their product XY is approximately equal to M. The objective function we minimize is the root-mean-square error.

This code contains only the "balanced strata" scheme for parallelization, which gives the best results compared to other schemes we've used. If you want to experiment with the other schemes used in our research, please contact us.

## Application -- MovieLens movie ratings database
We have tested our matrix factorization algorithm with a real-life example, the [MovieLens data](http://grouplens.org/datasets/movielens/). In particular, we have used the 1M, 10M, and 20M datasets (after light preprocessing to make it compatible with the HAMSI input format).

The accompanying research paper contains the table of results (final rms errors) for fixed settings of the parameters. In addition to that, we have explored the parameter space in order to find smaller errors (a better approximation to the original matrix).

Using 50 as the latent dimension size and running the algorithm for 2000 seconds, our best results so far are:

|dataset|sigma|gamma|etaLB|rmse|
|-------|-----|----|-----|----|
|1M|500|0.4|0.02|0.502|
|10M|500|0.1|0.01|0.586|
|20M|1000|0.1|0.01|0.651|

Please see the following sections for more information about replicating these results.

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
## Files
The code consists of a single file, `hamsi_sharedmem.cpp`. The Movielens 1M, 10M and 20M data files used in the research paper are provided in the `data.zip` file. Note that these are not identical to the data files in the MovieLens web site; the data format has been changed to accomodate the input standard of HAMSI.

## Compiling
The code is written in C++. The OpenMP library and GSL (GNU Scientific Library) development files are required.

To compile on the command line:
`g++ -std=c++11 hamsi_sharedmem.cpp -lgsl -lgslcblas -fopenmp -O3 -o hamsi`

## Usage
The `hamsi` executable takes the following command line arguments.
```
hamsi <data file> [-p<number of threads>] [-l<latent dimension>] [-i<max iteration>]
[-t<max time>] [-s<seed>] [-g<gamma>] [-e<etaLB>] [-a<sigma>] [-m<memory size>]
```
or, with long option names:
```
hamsi <data file>
      [--nthreads=<number of threads>]
      [--latentdim=<latent dimension>]
      [--maxiters=<max iteration>]
      [--maxtime=<max time>]
      [--randomseed=<seed>]
      [--gamma=<gamma>]
      [--eta=<etaLB>]
      [--sigma=<sigma>]
      [--memory=<memory size>]
```

|Argument|Description|Default value|
|--------|-----------|-------------|
|data file|The text file containing the sparse matrix in the form described above.|Required|
|number of threads|Number of processor threads that will be used in the parallel computation.|1|
|latent dimension|The inner dimension k of the factor matrices, which have sizes m-by-k and k-by-n, respectively.|5|
|max iteration|Stop the computation after so many iterations.|1000|
|max time|Stop the computation after so many seconds of wallclock time.|100|
|seed|The seed for random number generation.|1453|
|gamma|Step size adjustment parameter|0.51|
|etaLB|Step size adjustment parameter|0.06|
|sigma|Initial value for the L-BFGS scaling factor|500|
|memory|Memory size for L-BFGS|5|

Examples:

`./hamsi 1M.dat -p4 -l50 -i5000 -t10 -s123`

`./hamsi 1M.dat --nthreads=4 --latentdim=50 --maxtime=10 --gamma=0.4 --eta=0.05`

Notes:

- In the first use of the data file, HAMSI generates a binary copy for more efficient computation and storage. It is stored in the same directory. If you do not erase it, the binary copy will be reused in the next run of the program.


## Output
HAMSI displays the details of iteration (such as time, iteration number and error) at each step on standard output (the screen). The resulting factor matrices are stored in files named `hamsi1.out` and `hamsi2.out` by default, in tabular text form. File `hamsi1.out` has `m` rows and `k` columns, and file `hamsi2.out` has `k` rows and `n` columns. Columns are separated with space, rows are separated with newline character.

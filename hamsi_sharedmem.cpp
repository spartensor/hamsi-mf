#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include "string.h"

#include <iomanip>
#include <ctime>
#include <random>
#include <vector>

#include <gsl/gsl_linalg.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_permutation.h>
#include <omp.h>
#include <getopt.h>

using namespace std;

mt19937 rgen;
normal_distribution<double> randn(0, 1);

// Write the resulting matrix factors into text files.
void output_results(gsl_vector* x, int size1, int size2, int latentsize)
{
	FILE *outfile;
	outfile = fopen("factor1.dat","w");
	for(int row=0; row<size1; row++){
		for(int col=0; col<latentsize; col++)
			fprintf(outfile, "%f ",x->data[row*latentsize + col]);
		fprintf(outfile,"\n");
	}
	fclose(outfile);
	outfile = fopen("factor2.dat","w");
	for(int row=0; row<latentsize; row++){
		for(int col=0; col<size2;col++)
			fprintf(outfile, "%f ",x->data[row + (size1+col)*latentsize]);
		fprintf(outfile,"\n");
	}
	fclose(outfile);
}
//this function initializes the given matrix m randomly
//by using randn defined above
void randi_gsl_matrix(gsl_matrix* m) {
  for (unsigned int i = 0; i < m->size1; i++) {
    for (unsigned int j = 0; j < m->size2; j++) {
      gsl_matrix_set (m, i, j, randn(rgen));
    }
  }
}

//these handle binary I/O for faster re-execution
int readBinaryData(FILE* bp, int **dim_cards, int** indices, double **vals, int* tensor_dim, int* nnz) {
  fread(tensor_dim, sizeof(int), 1, bp);
  (*dim_cards) = new int[*tensor_dim];
  fread(*dim_cards, sizeof(int), (size_t)(*tensor_dim), bp);

  fread(nnz, sizeof(int), 1, bp);

  (*vals) = new double[*nnz];
  fread(*vals, sizeof(double), (size_t)(*nnz), bp);
  cout << "values are read " << endl;
  (*indices) = (int*) malloc(sizeof(int) * (*nnz) * (*tensor_dim));
  fread((*indices), sizeof(int), (size_t)((*tensor_dim) * (*nnz)), bp);
  cout << "indices are read " << endl;
  return 1;
}

int writeBinaryData(FILE* bp, int *dim_cards, int* indices, double* vals, int tensor_dim, int nnz) {
  fwrite(&tensor_dim, sizeof(int), (size_t)1, bp);
  fwrite(dim_cards, sizeof(int), (size_t)(tensor_dim), bp);
  fwrite(&nnz, sizeof(int), 1, bp);
  fwrite(vals, sizeof(double), (size_t)(nnz), bp);
  cout << "values are written " << endl;
  fwrite(indices, sizeof(int), (size_t)(tensor_dim * nnz), bp);
  cout << "indices are written " << endl;
  return 1;
}

int main(int argc, char **argv) {
	char bfile[2048]; //binary data file name
	char filename[2048]; //original data file name
	int *indices, *temp_indices = nullptr; //stores the coordinates for the nonzeros in the matrix
	double *vals, *temp_vals = nullptr; //stores the values of the nonzeros in the matrix
	int* dim_cards; //stores the cardinalities for each matrix dimension
	int nnz, tensor_dim; 
	int memoryCounter = 0;

	FILE *bp, *f;

	/* Parameters and defaults*/
	int M = 5; // memory size
	double etaGD = 0.001;
	double etaLB = 0.06;
	double gamma = 0.51;
	double toma = 500;
	double validityThreshold = 1.0e-6;   //to avoid weird moves
	int NT = 1; //number of threads that will be used for parallelization
	int LDIM = 5; //latent dimension - the inner dimension for factorization
	int EPOCHS = 1000; // number of maximum outer iterations
	int MAX_TIME = 100; //maximum time allowed for factorization in seconds
	long int my_seed = 1453; // the random seed
	char outfile[250] = "factor";
	if (argc == 1) {
		cout << "Usage: " 
		<< argv[0] 
		<< "[-p<number of threads>] "
		<< "[-l<latent dim>] [-i<max. number of iterations>] "
		<< "[-t<max time>] [-s<random seed>] "
		<< "[-g<gamma>] [-e<etaLB>] "
		<< "[-a<toma>] [-m<memory size>] "
		<< "[-o<output file name>]" << endl;
	return 0;
	}
	
	/* Parse command-line arguments */
	strcpy(filename,argv[1]);
	{
		static const char *optString = "g::m::a::e::p::l::i::t::s::o::";
		static struct option long_options[] =
		{
			{"gamma", optional_argument, NULL, 'g'},
			{"memory",optional_argument, NULL, 'm'},
			{"toma",optional_argument, NULL, 'a'},
			{"eta", optional_argument, NULL, 'e'},
			{"nthreads", optional_argument, NULL, 'p'},
			{"latentdim", optional_argument, NULL, 'l'},
			{"maxiters", optional_argument, NULL, 'i'},
			{"maxtime", optional_argument, NULL, 't'},
			{"randomseed", optional_argument, NULL, 's'},
			{"output", optional_argument, NULL, 'o'},
			//{"help", no_argument, NULL, 'h'},
			{0,0,0,0}
		};
		/* getopt_long stores the option index here. */
		int option_index = 0;
		int opt = getopt_long( argc, argv, optString, long_options, &option_index );
		while (opt != -1)
		{
		  switch (opt)
		  {
			  case 'g': gamma =atof(optarg); break;
			  case 'm': M = atoi(optarg); break;
			  case 'a': toma = atof(optarg); break;
			  case 'e': etaLB = atof(optarg); break;
			  case 'p': NT = atoi(optarg); break;
			  case 'l': LDIM = atoi(optarg); break;
			  case 'i': EPOCHS = atoi(optarg); break;
			  case 't': MAX_TIME = atoi(optarg); break;
			  case 's': my_seed = atol(optarg); break;
			  case 'o': strcpy(outfile, optarg); break;
		  }
		  opt = getopt_long( argc, argv, optString, long_options, &option_index );
		}

	}
  cout << "etaGD " << etaGD << " -  etaLB: " << etaLB << " - gamma:" << gamma << endl;
  cout << "Random seed is " << my_seed << endl;

  sprintf(bfile, "%s.bin", filename);
  bp = fopen(bfile, "rb");
  if(bp != NULL) { /* read from binary */
    cout << "reading the data from binary file...\n";
    if(readBinaryData(bp, &dim_cards, &indices, &vals, &tensor_dim, &nnz) == -1) {
      cout << "error reading the factor graph in binary format\n";
      fclose(bp);
      return -1;
    }
    fclose(bp);
  } else { /* read from the original data file and create binary */
    cout<< "reading the data from " << filename << "...\n";
    if((f = fopen(filename,"r")) == NULL) {
      cout << "Invalid file\n";
      return 0;
    }
    
    fscanf(f, "%d\n", &tensor_dim);
    dim_cards = (int*) malloc(sizeof(int) * tensor_dim);
    for(int i = 0; i < tensor_dim; i++) fscanf(f, "%d", dim_cards + i);
    fscanf(f, "%d", &nnz);
    
    indices = new int[nnz * tensor_dim];
    vals = new double[nnz];
    
    for (int i = 0; i < nnz; i++) {
      for(int j = 0; j < tensor_dim; j++) {
	fscanf(f, "%d\t", indices + (i * tensor_dim) + j);
	indices[i * tensor_dim + j]--;
      }
      fscanf(f, "%lf\n", vals + i);
    }
    cout << "The file is read" << endl;

    cout  << "writing to binary format..." << endl;
    if (f !=stdin) fclose(f);
    bp = fopen(bfile, "wb");
    if(bp != NULL) {
      if(writeBinaryData(bp, dim_cards, indices, vals, tensor_dim, nnz)  == -1) {
	cout << "error writing to graph in binary format\n";
	fclose(bp);
	return -1;
      }
      fclose(bp);
    }
  }
  cout << "The data dimension is " << tensor_dim << ". There are " << nnz << " data points" << endl;
  for(int i = 0; i < tensor_dim; i++) {
    cout << "\tdimension " << i << " is " << dim_cards[i] << endl;
  }
  
	omp_set_num_threads(NT);

  cout << "Memory size " << M << " and latent dimension is " << LDIM << endl;
  cout << "Maximum allowed time is " << MAX_TIME << " seconds" << endl;
  
  //these arrays will be used for permutation where neccessary
  temp_indices = new int[nnz * tensor_dim];
  temp_vals = new double[nnz];
  
  int NO_CHUNK = pow(NT, tensor_dim - 1); //the strata size and the number of stratas/chunks is determined w.r.t. the number of threads
  cout << "Number of strata is " << NO_CHUNK << endl;

  //these will be used to reorganize the data points in the matrix for stratification
  int* tids = new int[nnz]; 
  int* strataPtrs = new int[NO_CHUNK * (NT + 1)]; //stores the start index for each strata block
  memset(strataPtrs, 0, sizeof(int) * NO_CHUNK * (NT + 1));

  //this is chunk ptrs, we will permute the nonzeros
  //and store them in a form where each chunk's nnzs
  //are adjacent. this cptrs will store the first nnz's location
  //of each chunk. (the last entry is nnz)
  int* cptrs = new int[NO_CHUNK + 1];
  memset(cptrs, 0, sizeof(int) * (NO_CHUNK + 1));

  //this is the chunk id of nnz
  int* nnz_chunk = new int[nnz];

  //we will do balancing and find stratification lines for each dimension
  int* dimCardSum = new int[tensor_dim + 1]; //prefix sum of dim_cards
  memcpy(dimCardSum + 1, dim_cards, sizeof(int) * tensor_dim);
  dimCardSum[0] = 0; for(int d = 0; d < tensor_dim; d++) dimCardSum[d+1] += dimCardSum[d];
  
  int sumDimCard = 0; //sum of dimension cardinalities
  for(int d = 0; d < tensor_dim; d++) sumDimCard += dim_cards[d];
  int* noNnzDim = new int[sumDimCard];
  memset(noNnzDim, 0, sizeof(int) * sumDimCard);
  
  //lets compute how many nnz we have at each dimension
  for(int i = 0; i < nnz; i++) {
    for(int d = 0; d < tensor_dim; d++) {
      int loc = dimCardSum[d] + indices[i * tensor_dim + d];
      noNnzDim[loc]++;
    }
  }
  
  //now find the borders for each block
  int* balancers = new int[tensor_dim * (NT + 1)];
  int desired_load = ceil((nnz * 1.0f) / NT);
  for(int d = 0; d < tensor_dim; d++) {
    int current_load = 0;
    int current_block = 1;
    
    balancers[d * (NT + 1)] = 0; //the first is always 0
    for(int i = 0; i < dim_cards[d]; i++) { 
      int prev_load = current_load;
      current_load += noNnzDim[dimCardSum[d] + i];
      
      if(current_load >= desired_load) { //we are at the border; just decide to include the next one or not
	if(prev_load > 0 && (current_load - desired_load > desired_load - prev_load)) { //end the current block (guaranteed to be non-empty): i will go to the next block
	  balancers[d * (NT + 1) + current_block] = i; //the next one will start with i 
	  current_load = noNnzDim[dimCardSum[d] + i];
	} else {
	  balancers[d * (NT + 1) + current_block] = i+1; //i is in this block so next block should start with the next row/column etc. 
	  current_load = 0;
	}
	current_block++;
	if(current_block == NT) {
	  break;
	}
      }
    }
  
    for(int i = current_block; i <= NT; i++) {
      balancers[d * (NT + 1) + i] = dim_cards[d];
    }
    
    cout << " -- balanced ptrs: ";
    for(int i = 0; i <= NT; i++) {
      cout << balancers[d * (NT+1) + i] << " "; 
    }
    cout << endl;
  }
  
  //lets find the strata ids for each data point
  for(int i = 0; i < nnz; i++) {
    int tid = 0;
    int strata_id = 0; //for stratas, strata ids define the chunks 
    
    for(int d = 0; d < tensor_dim; d++) {
      int block_coord = NT - 1;
      while(indices[i * tensor_dim + d] < balancers[d * (NT + 1) + block_coord]) {
	block_coord--;
      }

      if(d == 0) {
	tid = block_coord; 
      } else {
	int blockid = (tid - block_coord + NT) % NT; 
	strata_id += blockid * pow(NT, tensor_dim - d - 1);
      }
    }

    if(strata_id >= NO_CHUNK) {cout << "invalid chunk id " << strata_id << " " << NO_CHUNK << endl; return 1;};
    if(tid >= NT) {cout << "invalid thread id " << tid << " " << NT << endl; return 1;};
    
    nnz_chunk[i] = strata_id;
    tids[i] = tid;
    strataPtrs[(strata_id * (NT + 1)) + tid + 1]++;
    cptrs[nnz_chunk[i] + 1]++;
  }

  //this code permutes the nnzs in their strata order
  //and stores the permutation in cids array
  //the cptrs pointers are indices of this cids array
  for(int i = 0; i < NO_CHUNK; i++) {
    cptrs[i+1] += cptrs[i];
  }

  int *cids = new int[nnz];
  for(int i = 0; i < nnz; i++) {
    cids[cptrs[nnz_chunk[i]]++] = i;
  }
  for(int i = NO_CHUNK; i > 0; i--) {
    cptrs[i] = cptrs[i-1];
  }
  cptrs[0] = 0;
  
  //for stratification we need to divide each strata into NT sub-chunks for parallelization
  //i.e., we need to set the values of strataPtrs.
  //copies cids to a temporary place
  memcpy(temp_indices, cids, sizeof(int) * nnz);
  
  for(int t = 0; t < NO_CHUNK; t++) { //for each strata
    //this is the pointer array for this strata
    int *strataPtr = strataPtrs + (t * (NT + 1));

    //this is the region we repermute with respect to 
    //the thread we will use for this nonzero
    int *local_cids = cids + cptrs[t];
    //the initial values are coming from above
    //stratePtr[i+1] is the number of nnzs in this chunk
    //that will be processed by thread i (not i + 1 for compactness)
    for(int i = 0; i < NT; i++) {
      strataPtr[i+1] += strataPtr[i];
    }

    for(int i = cptrs[t]; i < cptrs[t+1]; i++) {
      int nz = temp_indices[i];
      local_cids[strataPtr[tids[nz]]++] = nz;
    }
    for(int i = NT; i > 0; i--) {
      strataPtr[i] = cptrs[t] + strataPtr[i-1];
    }
    strataPtr[0] = cptrs[t];

    /*cout << "Strata " << t << " has " << cptrs[t+1] - cptrs[t] << " nnzs: ";
    for(int i = 0; i < NT; i++) {
      cout << strataPtr[i+1] - strataPtr[i] << " "; 
    }
    cout << endl;*/
  }

  int chunkPerm[NO_CHUNK + 1];
  for(int i = 0; i < NO_CHUNK; i++) {
    chunkPerm[i] = i;
  }
  chunkPerm[NO_CHUNK] = 0;

  //we use EPOCHS as the number of inner iterations
  EPOCHS *= NO_CHUNK;
  cout << "Maximum number of iterations is " << EPOCHS << endl;

  //now permute the nonzeros w.r.t. their chunk ids; first find the nnz_perm
  //also find chunkPtrs array this is already computed as cids and cptrs
  int nnzPerm[nnz];

  //the chunk perm array stores the chunks to be processed
  //it does not change for the inc_det versions 
  int chunkPtrs[NO_CHUNK + 1];
  
  //we computed permutation and the ptrs above
  //we also have strataPtrs ready 
  memcpy(nnzPerm, cids, sizeof(int) * nnz);
  memcpy(chunkPtrs, cptrs, sizeof(int) * (NO_CHUNK + 1));
 
  //first copy the original indices and values to a temporary location
  memcpy(temp_indices, indices, sizeof(int) * tensor_dim * nnz);
  memcpy(temp_vals, vals, sizeof(double) * nnz);
#pragma omp parallel for //then go over the nnzPerm array and insert the appropriate indices and value to the indices array back
  for(int i = 0; i < nnz; i++) {
    int pi = nnzPerm[i];   
    vals[i] = temp_vals[pi];
    for(int j = 0; j < tensor_dim; j++) {
      indices[i * tensor_dim + j] = temp_indices[pi * tensor_dim + j];
    }
  }

  //data is ready to process now allocate the memory that will be used
  int K = LDIM * (dim_cards[0] + dim_cards[1]);
  int bsize = K / NT;

  gsl_vector *x = gsl_vector_alloc(K);   // Current Solution
  gsl_vector *g = gsl_vector_alloc(K);   // Gradient

  gsl_matrix_view A = gsl_matrix_view_array (x->data, dim_cards[0], LDIM);
  gsl_matrix_view B = gsl_matrix_view_array (x->data+ dim_cards[0] * LDIM, dim_cards[1], LDIM);
  gsl_matrix_view gA = gsl_matrix_view_array (g->data, dim_cards[0], LDIM);
  gsl_matrix_view gB = gsl_matrix_view_array (g->data+dim_cards[0]* LDIM, dim_cards[1], LDIM);

  gsl_vector_view A_row[dim_cards[0]];   gsl_vector_view B_col[dim_cards[1]]; 
  gsl_vector_view gA_row[dim_cards[0]];  gsl_vector_view gB_col[dim_cards[1]]; 
  
  for (int i = 0; i < dim_cards[0]; i++) {
    A_row[i] = gsl_matrix_row(&A.matrix, i);
    gA_row[i] = gsl_matrix_row(&gA.matrix, i);
  }
  
  for (int j = 0; j < dim_cards[1]; j++) {
    B_col[j] = gsl_matrix_row(&B.matrix, j);
    gB_col[j] = gsl_matrix_row(&gB.matrix, j);
  }

  gsl_vector_view gpar[NT];
  gsl_vector_view xpar[NT];
  
  int i;
  for (i = 0; i < NT-1; i++) {
    xpar[i] = gsl_vector_subvector(x, i*bsize, bsize);
    gpar[i] = gsl_vector_subvector(g, i * bsize, bsize);
  }
  xpar[i] = gsl_vector_subvector(x, i * bsize, K - (i*bsize));
  gpar[i] = gsl_vector_subvector(g, i * bsize, K - (i*bsize));

  int tH = 2 * M; //history matrix size
  gsl_vector *g_hat = gsl_vector_alloc(K);   // Temporary gradient update 
  
  gsl_vector *prev_x = gsl_vector_alloc (K);   // Prev Solution
  gsl_vector *prev_g = gsl_vector_alloc (K);
  
  gsl_vector *diff_x = gsl_vector_alloc (K);
  gsl_vector *diff_g = gsl_vector_alloc (K);
  gsl_vector *mul_vec = gsl_vector_alloc (K);
  
  gsl_matrix* W = gsl_matrix_alloc (tH, K);
  gsl_matrix_set_zero(W);
  
  gsl_matrix_view S = gsl_matrix_submatrix (W, 0 , 0, M, K);
  gsl_matrix_view Y = gsl_matrix_submatrix (W, M , 0, M, K);

  gsl_matrix* N = gsl_matrix_alloc (tH, tH);
  gsl_matrix_set_zero(N);

  gsl_vector* WTg = gsl_vector_alloc (tH);
  gsl_vector_view WTg_second =  gsl_vector_subvector (WTg, M, M);
  gsl_vector* NWTg = gsl_vector_alloc (tH);
  gsl_vector_view NWTg_second =  gsl_vector_subvector (NWTg, M, M);

  gsl_matrix_view N11 = gsl_matrix_submatrix (N, 0 , 0, M, M);
  gsl_matrix_view N12 = gsl_matrix_submatrix (N, 0 , M, M, M);
  gsl_matrix_view N21 = gsl_matrix_submatrix (N, M, 0, M, M);

  gsl_matrix* R = gsl_matrix_alloc (M, M);
  gsl_matrix* R_save = gsl_matrix_alloc (M, M);
  gsl_vector_view R_save_rows[M];
  for(int j = 0; j < M; j++) R_save_rows[j] = gsl_matrix_row(R_save, j);
  gsl_vector_view R_save_cols[M];
  for(int j = 0; j < M; j++) R_save_cols[j] = gsl_matrix_column(R_save, j);

  gsl_matrix* C = gsl_matrix_alloc (M, M);
  gsl_matrix* C_save = gsl_matrix_alloc (M, M);
  gsl_vector_view C_save_rows[M];
  for(int j = 0; j < M; j++) C_save_rows[j] = gsl_matrix_row(C_save, j);
  gsl_vector_view C_save_cols[M];
  for(int j = 0; j < M; j++) C_save_cols[j] = gsl_matrix_column(C_save, j);

  gsl_matrix* Cp = gsl_matrix_alloc (M, M);
  gsl_vector_view D = gsl_matrix_diagonal(R);
  gsl_vector_view Cdiag = gsl_matrix_diagonal(C);                                            
  gsl_permutation* p = gsl_permutation_alloc(M);

  //for parallel computation
  gsl_matrix_view Wpar[NT];
  gsl_vector* WTgpar[NT];
  gsl_matrix_view Spar[NT];
  gsl_matrix_view Ypar[NT];

  gsl_vector_view g_hatpar[NT];
  gsl_vector_view prev_gpar[NT];
  gsl_vector_view diff_gpar[NT];
  gsl_vector_view mul_vecpar[NT];

  gsl_vector_view diff_xpar[NT];
  gsl_vector_view prev_xpar[NT];
  gsl_matrix* RCpar[NT];
  
  gsl_vector_view Sparrows[NT][M];  
  gsl_vector_view Yparrows[NT][M];  

  gsl_vector* RM1[NT];

  for (i = 0; i < NT-1; i++) {
    prev_xpar[i] = gsl_vector_subvector(prev_x, i * bsize, bsize);
    prev_gpar[i] = gsl_vector_subvector(prev_g, i * bsize, bsize);

    g_hatpar[i] = gsl_vector_subvector(g_hat, i * bsize, bsize);

    diff_xpar[i] = gsl_vector_subvector(diff_x, i * bsize, bsize);
    diff_gpar[i] = gsl_vector_subvector(diff_g, i * bsize, bsize);
    mul_vecpar[i] = gsl_vector_subvector(mul_vec, i * bsize, bsize);

    Wpar[i] = gsl_matrix_submatrix(W, 0, i * bsize, 2 * M, bsize);

    Spar[i] = gsl_matrix_submatrix(&S.matrix, 0, i * bsize, M, bsize);
    for(int j = 0; j < M; j++) {
      Sparrows[i][j] = gsl_matrix_row(&Spar[i].matrix, j);
    }
    Ypar[i] = gsl_matrix_submatrix(&Y.matrix, 0, i * bsize, M, bsize);
    for(int j = 0; j < M; j++) {
      Yparrows[i][j] = gsl_matrix_row(&Ypar[i].matrix, j);
    }    

    WTgpar[i] = gsl_vector_alloc(tH);
    RCpar[i] = gsl_matrix_alloc(M, M);

    RM1[i] = gsl_vector_alloc(M);
  }

  int remain = K - ((NT - 1) * bsize);
  prev_xpar[i] = gsl_vector_subvector(prev_x, i * bsize, remain);
  prev_gpar[i] = gsl_vector_subvector(prev_g, i * bsize, remain);

  g_hatpar[i] = gsl_vector_subvector(g_hat, i * bsize, remain);;

  diff_xpar[i] = gsl_vector_subvector(diff_x, i * bsize,remain);
  diff_gpar[i] = gsl_vector_subvector(diff_g, i * bsize,remain);
  mul_vecpar[i] = gsl_vector_subvector(mul_vec, i * bsize, remain);

  Wpar[i] = gsl_matrix_submatrix(W, 0, i * bsize, 2 * M, remain);

  Spar[i] = gsl_matrix_submatrix(&S.matrix, 0, i * bsize, M, remain);
  for(int j = 0; j < M; j++) {
    Sparrows[i][j] = gsl_matrix_row(&Spar[i].matrix, j);
  }
  Ypar[i] = gsl_matrix_submatrix(&Y.matrix, 0, i * bsize, M, remain);
  for(int j = 0; j < M; j++) {
    Yparrows[i][j] = gsl_matrix_row(&Ypar[i].matrix, j);
  }    

  WTgpar[i] = gsl_vector_alloc(tH);
  RCpar[i] = gsl_matrix_alloc(M, M);
  RM1[i] = gsl_vector_alloc(M);

  int signum;
  // Precompute and store views

  gsl_vector_view S_row[M];
  for (int i = 0; i < M; i++) { 
    S_row[i] = gsl_matrix_row(&S.matrix, i);
  }

  gsl_vector_view Y_row[M];
  for (int i = 0; i < M; i++) { 
    Y_row[i] = gsl_matrix_row(&Y.matrix, i);
  }


 
  //-------------------------------------------------------------------------
  // Random Initialization
  rgen.seed(my_seed);

  randi_gsl_matrix(&A.matrix);
  randi_gsl_matrix(&B.matrix);
  cout << "Start - x-norm: " << gsl_blas_dnrm2(x) << endl; 
  //-------------------------------------------------------------------------

  //Here we go
  int e = 0;
  double total_time = 0;
  for (; e < EPOCHS; e++) {
    double start_time = omp_get_wtime();
    double betaLB = pow(etaLB * (e+1), gamma);
    double betaGD = pow(etaGD * (e+1), gamma);

    if(e % (NO_CHUNK + 1) == 0) {
      for(int i = 0; i < NO_CHUNK; i++) {
	chunkPerm[i] = chunkPerm[i+1];
      }
      chunkPerm[NO_CHUNK] = chunkPerm[0];
    }
    //****************************************************************************
    //reset gradient vector
    if(NT == 1) {
      gsl_vector_set_zero(g);
    } else {
#pragma omp parallel for schedule(static) 
      for(int i = 0; i < NT; i++) {
	gsl_vector_set_zero(&gpar[i].vector);
      }
    }

    //****************************************************************************
    //compute the gradient for the current strata
    int chunk = chunkPerm[e % (NO_CHUNK + 1)]; 
#pragma omp parallel for schedule(static)
    for(int strata_block = 0; strata_block < NT; strata_block++) {
      int ptrLoc = chunk * (NT + 1) + strata_block;
      int start = strataPtrs[ptrLoc];
      int end = strataPtrs[ptrLoc+1];

      for (int nz = start; nz < end; nz++) {
	int i = indices[nz * tensor_dim];
	int j = indices[nz * tensor_dim + 1];
	
	double res;
	gsl_blas_ddot(&A_row[i].vector, &B_col[j].vector, &res);
	
	double delta = (vals[nz]-res) / nnz;
	gsl_blas_daxpy(-delta, &A_row[i].vector, &gB_col[j].vector);
	gsl_blas_daxpy(-delta, &B_col[j].vector, &gA_row[i].vector);
      }
    }
    //****************************************************************************
    
    //****************************************************************************
    //at this point we have a solution-gradient pair: if we have the previous ones update S and Y
    if(e > 0) { //we don't have a previous one for the first iteration      
      //for single thread/strata variants this happens at every pass, otherwise
      if(NT == 1 || (e % (NO_CHUNK + 1) == NO_CHUNK)) { //we need to be at the end of the chunk set
	double sy = 0, yy = 0;
	if(NT == 1) {
	  gsl_vector_memcpy(diff_x, x);
	  gsl_vector_sub(diff_x, prev_x);
	  
	  gsl_vector_memcpy(diff_g, g);
	  gsl_vector_sub(diff_g, prev_g);
	  
	  gsl_blas_ddot (diff_x, diff_g, &sy);
	  gsl_blas_ddot (diff_g, diff_g, &yy);
	} else {
#pragma omp parallel for 
	  for(int i = 0; i < NT; i++) {
	    gsl_vector_memcpy(&diff_xpar[i].vector, &xpar[i].vector);
	    gsl_vector_sub(&diff_xpar[i].vector, &prev_xpar[i].vector);
	  }
#pragma omp parallel for 
	  for(int i = 0; i < NT; i++) {
	    gsl_vector_memcpy(&diff_gpar[i].vector, &gpar[i].vector);
	    gsl_vector_sub(&diff_gpar[i].vector, &prev_gpar[i].vector);
	  }

#pragma omp parallel for reduction(+ : sy)
 	  for(int i = 0; i < NT; i++) {
	    gsl_blas_ddot(&diff_xpar[i].vector, &diff_gpar[i].vector, &sy);
	  }
#pragma omp parallel for reduction(+ : yy)
 	  for(int i = 0; i < NT; i++) {
	    gsl_blas_ddot(&diff_gpar[i].vector, &diff_gpar[i].vector, &yy);
	  }
	}

	//this avoids weird jumps 
	if(sy / sqrt(yy) > validityThreshold) {
	  toma = min(toma, sy/yy);
	  //data is passed now update S and Y accordingly
	  int writeLoc = memoryCounter % M;
	  memoryCounter++;
	  if(NT == 1) {
	    gsl_vector_memcpy(&(S_row[writeLoc].vector), diff_x);
	    gsl_vector_memcpy(&(Y_row[writeLoc].vector), diff_g);
	  } else {
#pragma omp parallel for schedule(static)
	    for(int j = 0; j < NT; j++) {
	      gsl_vector_memcpy(&Sparrows[j][writeLoc].vector, &diff_xpar[j].vector);
	      gsl_vector_memcpy(&Yparrows[j][writeLoc].vector, &diff_gpar[j].vector);
	    }
	  }
	  
	  //if the history is full update N so we can have a better update next time
	  if(memoryCounter >= M) { //do these operations when the memory is full, and when the S and Y are updated
                                   ///we avoid when S and Y are not updated since the values will be the same. 
	    if(memoryCounter == M) {
	      //R is S^T x Y
	      if(NT == 1) {
		gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0f, &S.matrix, &Y.matrix, 0.0f, R);
	      } else {	
#pragma omp parallel for schedule(static)
		for(int i = 0; i < NT; i++) {
		  gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0f, &Spar[i].matrix, &Ypar[i].matrix, 0.0f, RCpar[i]);
		}
		
		for(int i = 1; i < NT; i++) {
		  gsl_matrix_add(RCpar[0], RCpar[i]);
		}
		gsl_matrix_memcpy(R, RCpar[0]);
	      }    
	      gsl_matrix_memcpy(R_save, R);
	    } else {
	      if(NT == 1) {
		  gsl_blas_dgemv(CblasNoTrans, 1.0f, &S.matrix, &Y_row[writeLoc].vector, 0.0f, RM1[0]);
		  gsl_vector_memcpy(&R_save_cols[writeLoc].vector, RM1[0]);
		  gsl_blas_dgemv(CblasNoTrans, 1.0f, &Y.matrix, &S_row[writeLoc].vector, 0.0f, RM1[0]);
		  gsl_vector_memcpy(&R_save_rows[writeLoc].vector, RM1[0]);
	      } else {
#pragma omp parallel for schedule(static)
		for(int i = 0; i < NT; i++) {
		  gsl_blas_dgemv(CblasNoTrans, 1.0f, &Spar[i].matrix, &Yparrows[i][writeLoc].vector, 0.0f, RM1[i]);
		}
		
		for(int i = 1; i < NT; i++) {
		  gsl_vector_add(RM1[0], RM1[i]);
		}
		gsl_vector_memcpy(&R_save_cols[writeLoc].vector, RM1[0]);
#pragma omp parallel for schedule(static)
		for(int i = 0; i < NT; i++) {
		  gsl_blas_dgemv(CblasNoTrans, 1.0f, &Ypar[i].matrix, &Sparrows[i][writeLoc].vector, 0.0f, RM1[i]);
		}
		for(int i = 1; i < NT; i++) {
		  gsl_vector_add(RM1[0], RM1[i]);
		}
		gsl_vector_memcpy(&R_save_rows[writeLoc].vector, RM1[0]);
	      }
	      gsl_matrix_memcpy(R, R_save);
	    }	    

	    for (int i = 0; i <= writeLoc; i++) {
	      for (int j = 0; j < i; j++) {
		gsl_matrix_set(R, i, j, 0.0f);
	      }
	      for (int j = writeLoc + 1; j < M; j++) {
		gsl_matrix_set(R, i, j, 0.0f);
	      }
	    }
	    for (int i = writeLoc + 1; i < M; i++) {
	      for (int j = writeLoc + 1; j < i; j++) {
		gsl_matrix_set(R, i, j, 0.0f);
	      }
	    }

            if(memoryCounter == M) {
	      //C is (toma x Y^T x Y + D) 
	      if(NT == 1) {
		gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0f, &Y.matrix, &Y.matrix, 0.0f, C); 
	      } else {	
#pragma omp parallel for schedule(static)
		for(int i = 0; i < NT; i++) {
		  gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0f, &Ypar[i].matrix, &Ypar[i].matrix, 0.0f, RCpar[i]);
		}
		
		gsl_matrix_memcpy(C, RCpar[0]);
		for(int i = 1; i < NT; i++) {
		  gsl_matrix_add(C, RCpar[i]);
		}
	      }
	      gsl_matrix_memcpy(C_save, C);
	      gsl_matrix_scale(C, toma);
	    }  else {
	      if(NT == 1) {
		gsl_blas_dgemv(CblasNoTrans, 1.0f, &Y.matrix, &Y_row[writeLoc].vector, 0.0f, RM1[0]);
	      } else { 
#pragma omp parallel for schedule(static)
		for(int i = 0; i < NT; i++) {
		  gsl_blas_dgemv(CblasNoTrans, 1.0f, &Ypar[i].matrix, &Yparrows[i][writeLoc].vector, 0.0f, RM1[i]);
		}
		for(int i = 1; i < NT; i++) {
		  gsl_vector_add(RM1[0], RM1[i]);
		}
	      }
	      gsl_vector_memcpy(&C_save_rows[writeLoc].vector, RM1[0]);
	      gsl_vector_memcpy(&C_save_cols[writeLoc].vector, RM1[0]);
	      gsl_matrix_memcpy(C, C_save);
	      gsl_matrix_scale(C, toma);
	    }

	    gsl_vector_add (&Cdiag.vector, &D.vector);
	    //this computes N21 = -R^-1
	    gsl_linalg_LU_decomp (R, p, &signum);
	    gsl_linalg_LU_invert (R, p, &N21.matrix);
	    gsl_matrix_scale (&N21.matrix, -1.0f);
	    	    
	    //this computes N12 = -R^-T
	    gsl_matrix_transpose_memcpy (&N12.matrix, &N21.matrix);
	    
	    //this is N11 = N12 * C * N21
	    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0f, &N12.matrix, C, 0.0f, Cp);
	    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0f, Cp, &N21.matrix, 0.0f, &N11.matrix);	    													 
	  }
	}
      }
    }

    //now store g and x as previous for stochastic this happens only if we are at 
    //the beginning of the chunk set
    if(NT == 1 || e % (NO_CHUNK + 1) == 0) { //otherwise do it for the first chunk of the current chunk set
      //store current g and x in previous arrays
      if(NT == 1) { 
	gsl_vector_memcpy(prev_x, x);
	gsl_vector_memcpy(prev_g, g);
      } else {
#pragma omp parallel for schedule(static) 
      	for(int i = 0; i < NT; i++) {
	  gsl_vector_memcpy(&prev_xpar[i].vector, &xpar[i].vector);
	  gsl_vector_memcpy(&prev_gpar[i].vector, &gpar[i].vector);
	}
      }
    }
    //****************************************************************************

 
    //****************************************************************************
    //now do the update with the current gradient
    if(memoryCounter < M) { //we don't have enough information to do LBFGS updates; go with GD
      if(NT == 1) {
	gsl_blas_daxpy((-1.0f/betaGD), g, x); // Update                                                                                                                                  
      } else {
#pragma omp parallel for schedule(static)
	for(int i = 0; i < NT; i++) {
	  gsl_blas_daxpy((-1.0f/betaGD), &gpar[i].vector, &xpar[i].vector); // Update                                                                                                   
	}
      }
    } else {     
      //now update solution: this happens at every iteration
      if(NT == 1) {
	gsl_vector_memcpy(g_hat, g);
	gsl_vector_scale(g_hat, toma);
	gsl_blas_dgemv(CblasNoTrans, 1.0f, W, g, 0.0f, WTg);
	gsl_vector_scale(&WTg_second.vector, toma); 
	gsl_blas_dgemv(CblasNoTrans, 1.0f, N, WTg, 0.0f, NWTg);
	gsl_vector_scale(&NWTg_second.vector, toma); 
	gsl_blas_dgemv(CblasTrans, 1.0f, W, NWTg, 1.0f, g_hat); 
	gsl_blas_daxpy((-1.0f/betaLB), g_hat, x); // Update
      } else {
#pragma omp parallel for schedule(static)
	for(int i = 0; i < NT; i++) {
	  gsl_vector_memcpy(&g_hatpar[i].vector, &gpar[i].vector);
	  gsl_vector_scale(&g_hatpar[i].vector, toma);
	}

#pragma omp parallel for schedule(static)
	for(int i = 0; i < NT; i++) {
	  gsl_blas_dgemv(CblasNoTrans, 1.0f, &Wpar[i].matrix, &gpar[i].vector, 0.0f, WTgpar[i]);
	}
       	gsl_vector_memcpy(WTg, WTgpar[0]);
	for(int i = 1; i < NT; i++) {
	  gsl_vector_add(WTg, WTgpar[i]);
	}

	gsl_vector_scale(&WTg_second.vector, toma);       
	gsl_blas_dgemv(CblasNoTrans, 1.0f, N, WTg, 0.0f, NWTg);
	gsl_vector_scale(&NWTg_second.vector, toma); 

#pragma omp parallel for schedule(static)
	for(int i = 0; i < NT; i++) {
	  gsl_blas_dgemv(CblasTrans, 1.0f, &Wpar[i].matrix, NWTg, 1.0f, &g_hatpar[i].vector);  
	}

#pragma omp parallel for schedule(static)
	for(int i = 0; i < NT; i++) {
	  gsl_blas_daxpy((-1.0f/betaLB), &g_hatpar[i].vector, &xpar[i].vector); // Update
	}
      }
    }
    //****************************************************************************


    total_time += omp_get_wtime() - start_time;    
    if(total_time > MAX_TIME) {
      cout << "exiting due to time limit: " << total_time << " seconds passed " << endl;
      break;
    } 

#define DEBUG
    if (NT == 1 || (e+1) % (NO_CHUNK) == 0) {
      cout << "Data pass time: " << total_time << " in inner iteration " << e << endl ; 
#ifdef DEBUG
      double err = 0.0;
#pragma omp parallel for reduction(+ : err)
      for (int nz = 0; nz < nnz; nz++) {
	int i = indices[nz * tensor_dim];
	int j = indices[nz * tensor_dim + 1];
	double res;
	gsl_blas_ddot (&A_row[i].vector, &B_col[j].vector, &res);
	double tmp = vals[nz]-res;
	err += tmp*tmp;
     }
     if (err != err)
		break;
      cout << "Current error is: " << sqrt(err/nnz) << " in " << e << " iterations" << endl;
#endif
    }
  }

  double err = 0.0;
#pragma omp parallel for reduction(+ : err)
  for (int nz = 0; nz < nnz; nz++) {
    int i = indices[nz * tensor_dim];
    int j = indices[nz * tensor_dim + 1];
    double res;
    gsl_blas_ddot (&A_row[i].vector, &B_col[j].vector, &res);
    double tmp = vals[nz]-res;
    err += tmp*tmp;
  }
  cout << "Final error is:  " << sqrt(err/nnz) << " in " << e << " iterations" << endl;

  cout << "Total time is:  " << total_time << endl;
  
  output_results(x, dim_cards[0], dim_cards[1], LDIM);
  gsl_vector_free (x);
  gsl_vector_free (g);

  return 0;
}



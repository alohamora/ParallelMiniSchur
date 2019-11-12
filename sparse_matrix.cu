#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include "sparse_matrix.h"
#include "helper.h"
#define ll long long int
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

ll *MAJOR_SORTING_INDEX, *MINOR_SORTING_INDEX;

int cmp(const void *l, const void *r);
void permuteMatrix(sparse_matrix_t* A, ll permute[]);

__global__ void copy_matrix_kernel(ll *A_rows, ll *A_cols, float *A_vals, ll *s_rows, ll *s_cols, float *s_vals, ll *stInd, ll *cum, ll colL, int base){
    ll totalCols, index;
    totalCols = (blockIdx.x > 0) ? cum[blockIdx.x] - cum[blockIdx.x - 1] : cum[blockIdx.x];
    index = (blockIdx.x > 0) ? cum[blockIdx.x - 1] : 0;
    if(threadIdx.x == 0)    s_rows[blockIdx.x] = index;
    if(threadIdx.x < totalCols){
        s_cols[index + threadIdx.x] = A_cols[stInd[blockIdx.x] + threadIdx.x] - colL + base;
        s_vals[index + threadIdx.x] = A_vals[stInd[blockIdx.x] + threadIdx.x];
    }
}

void getSubMatrix(sparse_matrix_t* A, sparse_matrix_t *sub, sparse_matrix_t *d_A, sparse_matrix_t *d_sub, ll rowL, ll rowR, ll colL, ll colR, int cpy2cpu){
    ll *stInd, *cum, *d_stInd, *d_cum;
    ll l, r, maxCols = 0;
    d_sub->n = rowR - rowL + 1;
    d_sub->m = colR - colL + 1;
    d_sub->memtype = d_A->memtype;
    d_sub->descr = A->descr;
    d_sub->order = A->order;
    d_sub->index = A->index;
    stInd = (ll *)malloc(d_sub->n * sizeof(ll));
    cum = (ll *)malloc(d_sub->n * sizeof(ll));
    for(int i=rowL;i<=rowR;i++){
        l = lowerBound(A->cols, A->rows[i - A->index], (i == A->n - (1 - A->index)) ? A->nnz - 1 : A->rows[i - A->index + 1] - 1, colL);
        r = upperBound(A->cols, A->rows[i - A->index], (i == A->n - (1 - A->index)) ? A->nnz - 1 : A->rows[i - A->index + 1] - 1, colR);
        stInd[i-rowL] = l;
        cum[i-rowL] = (i > rowL) ? cum[i-rowL-1] + MAX(r-l, 0) : MAX(r-l, 0);
        maxCols = MAX(r-l, maxCols);
    }
    d_sub->nnz = cum[rowR - rowL];
    sparseMatrixMalloc(d_sub);
    cudaMalloc((void **)&(d_stInd), d_sub->n * sizeof(ll));
    cudaMalloc((void **)&(d_cum), d_sub->n * sizeof(ll));
    cudaMemcpy(d_stInd, stInd, d_sub->n * sizeof(ll), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cum, cum, d_sub->n * sizeof(ll), cudaMemcpyHostToDevice);
    copy_matrix_kernel<<<d_sub->n, maxCols>>>(d_A->rows, d_A->cols, d_A->vals, d_sub->rows, d_sub->cols, d_sub->vals, d_stInd, d_cum, colL, A->index);
    if(cpy2cpu)
        sparseMatrixCopy(d_sub, sub, CPU);
}

void sparseMatrixCopy(sparse_matrix_t* A, sparse_matrix_t* B, enum memory_type mem){
    B->n = A->n;
    B->m = A->m;
    B->nnz = A->nnz;
    B->descr = A->descr;
    B->order = A->order;
    B->index = A->index;
    B->memtype = mem;
    sparseMatrixMalloc(B);
    if(A->memtype == CPU && B->memtype == GPU){
        if(A->descr == CSR)    cudaMemcpy(B->rows, A->rows, (A->n+1) * sizeof(ll), cudaMemcpyHostToDevice);
        else    cudaMemcpy(B->rows, A->rows, A->nnz * sizeof(ll), cudaMemcpyHostToDevice);
        cudaMemcpy(B->cols, A->cols, A->nnz * sizeof(ll), cudaMemcpyHostToDevice);
        cudaMemcpy(B->vals, A->vals, A->nnz * sizeof(float), cudaMemcpyHostToDevice);
    }
    else if(A->memtype == GPU && B->memtype == CPU){
        if(A->descr == CSR)    cudaMemcpy(B->rows, A->rows, (A->n+1) * sizeof(ll), cudaMemcpyDeviceToHost);
        else    cudaMemcpy(B->rows, A->rows, A->nnz * sizeof(ll), cudaMemcpyDeviceToHost);
        cudaMemcpy(B->cols, A->cols, A->nnz * sizeof(ll), cudaMemcpyDeviceToHost);
        cudaMemcpy(B->vals, A->vals, A->nnz * sizeof(float), cudaMemcpyDeviceToHost);
    }
}

void sparseMatrixMalloc(sparse_matrix_t* A){
    if(A->memtype == CPU){
        if(A->descr == COO)    A->rows = (ll *)malloc(A->nnz * sizeof(ll));
        else    A->rows = (ll *)malloc((A->n+1) * sizeof(ll));
        A->cols = (ll *)malloc(A->nnz * sizeof(ll));
        A->vals = (float *)malloc(A->nnz * sizeof(float));
    }
    else{
        if(A->descr == COO)    cudaMalloc((void **)&(A->rows), A->nnz * sizeof(ll));
        else    cudaMalloc((void **)&(A->rows), (A->n+1) * sizeof(ll));
        cudaMalloc((void **)&(A->cols), A->nnz * sizeof(ll));
        cudaMalloc((void **)&(A->vals), A->nnz * sizeof(float));
    }
}

void sparseMatrixFree(sparse_matrix_t* A){
    if(A->memtype == CPU){
        free(A->rows);
        free(A->cols);
        free(A->vals);
    }
    else{
        cudaFree(A->rows);
        cudaFree(A->cols);
        cudaFree(A->vals);
    }
}

void coo2csr(sparse_matrix_t* A){
    ll *permute, *rowIndex, row;
    if(A->order == COLUMN_MAJOR){
        MAJOR_SORTING_INDEX = A->rows;
        MINOR_SORTING_INDEX = A->cols;
        permute = (ll *)malloc(A->nnz * sizeof(ll));
        for(ll i=0;i<A->nnz;i++)   permute[i] = i;
        qsort(permute, A->nnz, sizeof(ll), cmp);
        permuteMatrix(A, permute);
    }
    A->descr = CSR;
    A->order = ROW_MAJOR;
    rowIndex = (ll *)malloc((A->n + 1) * sizeof(ll));
    rowIndex[0] = 0;
    row = (ll)A->index;
    for(ll i=0;i<A->nnz;i++){
        if(A->rows[i] != row){
            while(row != A->rows[i])    rowIndex[(++row) - A->index] = i; 
        }
    }
    rowIndex[A->n] = A->nnz;
    free(A->rows);
    A->rows = rowIndex;
}

void printMatrix(sparse_matrix_t* A){
    if(A->descr == COO)
        for(ll i=0;i<A->nnz;i++)    printf("%lld %lld %f\n", A->rows[i], A->cols[i], A->vals[i]);
    else{
        ll row = A->index, ind = 1;
        for(ll i=0;i<A->nnz;i++){
            while(i == A->rows[ind]){   ind++;  row++;  }
            printf("%lld %lld %f\n", row, A->cols[i], A->vals[i]);
        }
    }
}

int cmp(const void *l, const void *r){
    ll lind, rind;
    lind = *(ll *)l;
    rind = *(ll *)r;
    if(MAJOR_SORTING_INDEX[lind] < MAJOR_SORTING_INDEX[rind])   return -1;
    else if(MAJOR_SORTING_INDEX[lind] == MAJOR_SORTING_INDEX[rind]) return (MINOR_SORTING_INDEX[lind] < MINOR_SORTING_INDEX[rind]) ? -1 : 1;
    return 1;
}

void permuteMatrix(sparse_matrix_t* A, ll *permute){
    ll *permutedRows, *permutedCols;
    float *permutedVals;
    permutedVals = (float *)malloc(A->nnz * sizeof(float));
    permutedRows = (ll *)malloc(A->nnz * sizeof(ll));
    permutedCols = (ll *)malloc(A->nnz * sizeof(ll));
    for(ll i=0;i<A->nnz;i++){
        permutedRows[i] = A->rows[permute[i]];
        permutedCols[i] = A->cols[permute[i]];
        permutedVals[i] = A->vals[permute[i]];
    }
    for(ll i=0;i<A->nnz;i++){
        A->rows[i] = permutedRows[i];
        A->cols[i] = permutedCols[i];
        A->vals[i] = permutedVals[i];
    }
    free(permutedCols);
    free(permutedRows);
    free(permutedVals);
}
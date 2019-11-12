#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include<cusolverSp.h>
#include<cusparse.h>
#include "sparse_matrix.h"
#include "mini_schur.h"
#define ll long long int

cusparseMatDescr_t descrA;      
cusolverSpHandle_t solver_handle;

int solveLU(int N, int nnz, float* h_A, int* h_A_RowIndices, int* h_A_ColIndices, float* h_y, float* h_x);
void convert2cusparse(sparse_matrix_t* A);

__global__ void csr2dense(ll* rows, ll* cols, float* vals, ll n, float* dense){
    ll idx = rows[blockIdx.x] + threadIdx.x;
    if(idx < rows[blockIdx.x + 1])  dense[blockIdx.x + (cols[idx]*n)] = vals[idx];
}

void calculateMiniSchur(sparse_matrix_t* schur, sparse_matrix_t* D, sparse_matrix_t* L, sparse_matrix_t* U, sparse_matrix_t* G){
    float *d_y, *y, *x;
    sparse_matrix_t h_D;
    y = (float *)malloc((D->n * U->m) * sizeof(float));
    cudaMalloc((void **)&(d_y), (D->n * U->m) * sizeof(float));
    cudaMemset(d_y, 0.0f, (D->n * U->m) * sizeof(float));
    csr2dense<<<U->n, U->m>>>(U->rows, U->cols, U->vals, U->n, d_y);
    x = (float *)malloc((D->n * U->m) * sizeof(float));
    sparseMatrixCopy(D, &h_D, CPU);
    convert2cusparse(&h_D);
    cudaDeviceSynchronize();
    cudaMemcpy(y, d_y, (D->n * U->m) * sizeof(float), cudaMemcpyDeviceToHost);
    for(int i=0;i<U->m;i++)
        solveLU(h_D.n, h_D.nnz, h_D.vals, h_D.irows, h_D.icols, y + (i*(U->n)), x + (i*(U->n)));
}

int solveLU(int N, int nnz, float* h_A, int* h_A_RowIndices, int* h_A_ColIndices, float* h_y, float* h_x){
    int singularity;
    cusolverSpScsrlsvluHost(solver_handle, N, nnz, descrA, h_A, h_A_RowIndices, h_A_ColIndices, h_y, 0.000001, 0, h_x, &singularity);
    return singularity;
}

void convert2cusparse(sparse_matrix_t* A){
    A->irows = (int *)malloc((A->n+1) * sizeof(int));
    A->icols = (int *)malloc((A->nnz) * sizeof(int));
    for(int i=0;i<A->nnz;i++)  A->icols[i] = (int) A->cols[i];
    for(int i=0;i<=A->n;i++)  A->irows[i] = (int) A->rows[i] + 1;
}

void createHandles(){
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE);
    cusolverSpCreate(&solver_handle);
}
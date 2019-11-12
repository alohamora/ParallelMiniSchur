#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include<cusolverSp.h>
#include<cusparse.h>
#include "sparse_matrix.h"
#include "mini_schur.h"

cusparseMatDescr_t descrA;      
cusolverSpHandle_t solver_handle;

int solveLU(int N, int nnz, float* h_A, int* h_A_RowIndices, int* h_A_ColIndices, float* h_y, float* h_x);
void convert2cusparse(sparse_matrix_t* A, cusparse_matrix_csr* B);
void calculateMiniSchur(sparse_matrix_t* schur, sparse_matrix_t* D, sparse_matrix_t* L, sparse_matrix_t* U, sparse_matrix_t* G){
    float *d_y, *y, *x;
    sparse_matrix_t h_D, h_U;
    cusparse_matrix_csr  D_csr, U_csr;
    sparseMatrixCopy(D, &h_D, CPU);
    sparseMatrixCopy(U, &h_U, CPU);
    convert2cusparse(&h_U, &U_csr);
    convert2cusparse(&h_D, &D_csr);
    y = (float *)malloc(h_D.n * sizeof(float));
    cudaMalloc((void **)&(d_y), h_D.n * sizeof(ll));
    x = (float *)malloc(h_D.n * sizeof(float));
    for(int i=0;i<h_D.n;i++)    y[i] = i + 1;
    solveLU(h_D.n, h_D.nnz, h_D.vals, D_csr.rows, D_csr.cols, y, x);
}

int solveLU(int N, int nnz, float* h_A, int* h_A_RowIndices, int* h_A_ColIndices, float* h_y, float* h_x){
    int singularity;
    cusolverSpScsrlsvluHost(solver_handle, N, nnz, descrA, h_A, h_A_RowIndices, h_A_ColIndices, h_y, 0.000001, 0, h_x, &singularity);
    return singularity;
}

void convert2cusparse(sparse_matrix_t* A, cusparse_matrix_csr* B){
    B->rows = (int *)malloc((A->n+1) * sizeof(int));
    B->cols = (int *)malloc((A->nnz) * sizeof(int));
    for(int i=0;i<A->nnz;i++)  B->cols[i] = (int) A->cols[i];
    for(int i=0;i<=A->n;i++)  B->rows[i] = (int) A->rows[i] + 1;
}

void createHandles(){
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE);
    cusolverSpCreate(&solver_handle);
}
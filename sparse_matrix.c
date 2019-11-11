#include<stdio.h>
#include<stdlib.h>
#include "sparse_matrix.h"
#define ll long long int

int cmp(const void *l, const void *r);
void permuteMatrix(sparse_matrix_t* A, ll permute[]);

void sparseMatrixMalloc(sparse_matrix_t* A){
    if(A->memtype == CPU){
        if(A->descr == COO)    A->rows = (ll *)malloc(A->nnz * sizeof(ll));
        else    A->rows = (ll *)malloc(A->n * sizeof(ll));

        A->cols = (ll *)malloc(A->nnz * sizeof(ll));
        A->vals = (float *)malloc(A->nnz * sizeof(float));
    }
    else{
        // if(A->descr == COO)    cudaMalloc((void **)&(A->rows), A->nnz * sizeof(int));
        // else    cudaMalloc((void **)&(A->rows), A->n * sizeof(int));

        // cudaMalloc((void **)&(A->cols), A->nnz * sizeof(int));
        // cudaMalloc((void **)&(A->vals), A->nnz * sizeof(float));
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
    rowIndex = (ll *)malloc(A->n * sizeof(ll));
    rowIndex[0] = 0;
    row = (ll)A->index;
    for(ll i=0;i<A->nnz;i++){
        if(A->rows[i] != row){
            while(row != A->rows[i])    rowIndex[(++row) - A->index] = i; 
        }
    }
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
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include "sparse_matrix.h"
#include "mini_schur.h"
#define ll long long int

int main(int argc, char** argv){
    ll nparts, sizeG;
    ll stD, endD, stG, endG, subDsz, subGsz;
    clock_t t;
    double time_taken;
    sparse_matrix_t A, d_A, D, d_D, G, d_G, L, d_L, U, d_U;
    sparse_matrix_t *subD, *subL, *subU, *subG, *miniSchurs;
    if(argc != 7){
        printf("Incorrect no of parameters passed, command line format:\nmain [filename] [matrix_descr] [matrix_order] [matrix_index_format] [nparts] [sizeG]\n");
        exit(-1);
    }
    nparts = atoll(argv[5]);
    sizeG = atoll(argv[6]);
    miniSchurs = (sparse_matrix_t *)malloc(nparts * sizeof(sparse_matrix_t));
    subL = (sparse_matrix_t *)malloc(nparts * sizeof(sparse_matrix_t));
    subD = (sparse_matrix_t *)malloc(nparts * sizeof(sparse_matrix_t));
    subU = (sparse_matrix_t *)malloc(nparts * sizeof(sparse_matrix_t));
    subG = (sparse_matrix_t *)malloc(nparts * sizeof(sparse_matrix_t));
    FILE *lhs_input = fopen((const char *)argv[1],"r");
    if(!lhs_input){
            printf("file not found\n");
            exit(-1);
    }
    A.descr = atoi(argv[2]);
    A.order = atoi(argv[3]);
    A.index = atoi(argv[4]);
    A.memtype = CPU;
    fscanf(lhs_input, "%lld %lld %lld", &A.n, &A.m, &A.nnz);
    printf("Input matrix dimensions - (%lld, %lld)\n", A.n, A.m);
    printf("No of non zeros = %lld\n", A.nnz);
    sparseMatrixMalloc(&A);
    for(ll i=0;i<A.nnz;i++){
            fscanf(lhs_input, "%lld %lld %f", &A.rows[i],&A.cols[i],&A.vals[i]);
    }
    coo2csr(&A);
    t = clock();
    printf("Starting to build preconditioner...\n");
    sparseMatrixCopy(&A, &d_A, GPU);
    printf("Partitioning matrix into blocks...\n");
    getSubMatrix(&A, &D, &d_A, &d_D, 1, A.n - sizeG, 1, A.m - sizeG, 1);
    getSubMatrix(&A, &L, &d_A, &d_L, A.n - sizeG + 1, A.n, 1, A.m - sizeG, 1);
    getSubMatrix(&A, &U, &d_A, &d_U, 1, A.n - sizeG, A.m - sizeG + 1, A.m, 1);
    getSubMatrix(&A, &G, &d_A, &d_G, A.n - sizeG + 1, A.n, A.m - sizeG + 1, A.m, 1);
    sparseMatrixFree(&A);
    sparseMatrixFree(&d_A);

    stD = 1;
    subDsz = (D.n - (D.n % nparts))/nparts;
    endD = subDsz;
    stG = 1;
    subGsz = (G.n - (G.n % nparts))/nparts;
    endG = subGsz;
    createHandles();
    printf("Generating mini schur complements with no of parts = %lld\n", nparts);
    for(int i=0;i<nparts;i++){
        if(i == nparts - 1){
            endD = D.n;
            endG = G.n;
        }
        getSubMatrix(&D, NULL, &d_D, subD + i, stD, endD, stD, endD, 0);
        getSubMatrix(&L, NULL, &d_L, subL + i, stG, endG, stD, endD, 0);
        getSubMatrix(&U, NULL, &d_U, subU + i, stD, endD, stG, endG, 0);
        getSubMatrix(&G, NULL, &d_G, subG + i, stG, endG, stG, endG, 0);
        calculateMiniSchur(&miniSchurs[i], subD + i, subL + i, subU + i, subG + i);
        stD = endD + 1;
        endD += subDsz;
        stG = endG + 1;
        endG += subGsz;
        printf("Iteration %d completed...\n", i);
    }
    t = clock() - t;
    time_taken = ((double)t)/CLOCKS_PER_SEC;
    printf("Time used for building preconditioner: %lfs\n", time_taken);
    return 0;
}
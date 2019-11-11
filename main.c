#include<stdio.h>
#include<stdlib.h>
#include"sparse_matrix.h"
#define ll long long int

int main(int argc, char** argv){
    ll nparts, sizeG;
    sparse_matrix_t A;
    if(argc != 7){
        printf("Incorrect no of parameters passed, command line format:\nmain [filename] [matrix_descr] [matrix_order] [matrix_index_format] [nparts] [sizeG]\n");
        exit(-1);
    }
    nparts = atoll(argv[5]);
    sizeG = atoll(argv[6]);
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
    sparseMatrixMalloc(&A);
    for(ll i=0;i<A.nnz;i++){
            fscanf(lhs_input, "%lld %lld %f", &A.rows[i],&A.cols[i],&A.vals[i]);
    }
    coo2csr(&A);
    printMatrix(&A);
    // sparseMatrixCopy(&A, 1);
    // getSubmatrix(&A, &D, 1, A.n - sizeG, 1, A.m - sizeG);
    // getSubmatrix(&A, &L, A.n - sizeG + 1, A.n, 1, A.m - sizeG);
    // getSubmatrix(&A, &U, 1, A.n - sizeG, A.m - sizeG + 1, A.m);
    // getSubmatrix(&A, &G, A.n - sizeG + 1, A.n, A.m - sizeG + 1, A.m);

    // sparseMatrixFree(&A, 1);

    // stD = 1;
    // subDsz = (D.n - (D.n % nparts))/nparts;
    // endD = subDsz;
    // stG = 1;
    // subGsz = (G.n - (G.n % nparts))/nparts;
    // endG = subGsz;
    // for(int i=0;i<nparts;i++){
    //     if(i == nparts - 1){
    //         endD = D.n;
    //         endG = G.n;
    //     }
    //     getSubmatrix(&D, subD + i, stD, endD, stD, endD);
    //     getSubmatrix(&L, subL + i, stG, endG, stD, endD);
    //     getSubmatrix(&U, subU + i, stD, endD, stG, endG);
    //     getSubmatrix(&G, subG + i, stG, endG, stG, endG);
    //     // calculateMiniSchur(&miniSchurs[i], &subD, &subL, &subU, &subG);
    //     stD = endD + 1;
    //     endD += subDsz;
    //     stG = endG + 1;
    //     endG += subGsz;
    // }
    return 0;
}
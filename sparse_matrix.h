enum matrix_descr{
    CSR,
    COO
};

enum matrix_order{
    COLUMN_MAJOR,
    ROW_MAJOR
};

enum matrix_index{
    INDEX_BASE_ZERO,
    INDEX_BASE_ONE
};

enum memory_type{
    CPU,
    GPU
};

typedef struct sparse_matrix_t {
    enum matrix_descr descr;
    enum matrix_order order;
    enum matrix_index index;
    enum memory_type memtype;
    long long int n,m,nnz;
    long long int *rows, *cols;
    float *vals;
}sparse_matrix_t;

long long int *MAJOR_SORTING_INDEX, *MINOR_SORTING_INDEX;

#ifdef __cplusplus
extern "C" {
#endif
    void sparseMatrixMalloc(sparse_matrix_t* A);
    void coo2csr(sparse_matrix_t* A);
    void printMatrix(sparse_matrix_t* A);
#ifdef __cplusplus
}
#endif
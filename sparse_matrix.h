#define ll long long int
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
    ll n,m,nnz;
    ll *rows, *cols;
    float *vals;
}sparse_matrix_t;

ll *MAJOR_SORTING_INDEX, *MINOR_SORTING_INDEX;

#ifdef __cplusplus
extern "C" {
#endif
    void sparseMatrixMalloc(sparse_matrix_t* A);
    void printMatrix(sparse_matrix_t* A);
    void sparseMatrixCopy(sparse_matrix_t* A, sparse_matrix_t* B, enum memory_type mem);
    void sparseMatrixFree(sparse_matrix_t* A);
    void coo2csr(sparse_matrix_t* A);
    void getSubMatrix(sparse_matrix_t* A, sparse_matrix_t* sub, sparse_matrix_t *d_A, sparse_matrix_t *d_sub, ll rowL, ll rowR, ll colL, ll colR, int cpy2cpu);
#ifdef __cplusplus
}
#endif
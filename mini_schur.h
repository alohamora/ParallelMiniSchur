#ifdef __cplusplus
extern "C" {
#endif
    void calculateMiniSchur(sparse_matrix_t* schur, sparse_matrix_t* D, sparse_matrix_t* L, sparse_matrix_t* U, sparse_matrix_t* G);
    void createHandles();
#ifdef __cplusplus
}
#endif
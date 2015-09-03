package org.bitbucket.ioplus.clblas;



import static org.bitbucket.ioplus.clblas.BlasOrder.*;
import static org.bitbucket.ioplus.clblas.BlasTrans.*;

/**
 * Created by przemek on 22/08/15.
 */
public class Level3 {

    private Level3() {
    }

    /**
     * Matrix-matrix product of general rectangular matrices with float elements.
     * \( C \leftarrow \alpha A B + \beta C \)
     * \( C \leftarrow \alpha A^T B + \beta C \)
     * \( C \leftarrow \alpha A B^T + \beta C \)
     * \( C \leftarrow \alpha A^T B^T + \beta C \)
     *
     * @param order BlasOrder.ROW_MAJOR|COLUMN_MAJOR
     * @param transA How matrix A is to be transposed.
     * @param transB How matrix B is to be transposed.   Buffer object storing vector X
     * @param M      Number of rows in matrix A.
     * @param N      Number of columns in matrix B.
     * @param K      Number of columns in matrix A and rows in matrix B.
     * @param alpha  The factor of matrix A.
     * @param A      Buffer object storing matrix A.
     * @param lda    Leading dimension of matrix A. It cannot be less than K.
     * @param B      Buffer object storing matrix B.
     * @param ldb    Leading dimension of matrix B. It cannot be less than N.
     * @param beta   The factor of matrix C.
     * @param C      Buffer object storing matrix C.
     * @param ldc    Leading dimension of matrix C. It cannot be less than N.
     **/
    public static int sgemm(BlasOrder order,
                            BlasTrans transA,
                             BlasTrans transB,
                             int M,
                             int N,
                             int K,
                             float alpha,
                             final float[] A,
                             int lda,
                             final float[] B,
                             int ldb,
                             float beta,
                             float[] C,
                             int ldc) {
        return BLAS.sgemm(
                order.ordinal(),
                transA.ordinal(),
                transB.ordinal(),
                M, N, K,
                alpha, A, 0, lda,
                B, 0, ldb,
                beta, C, 0, ldc);
    }

}

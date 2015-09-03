package org.bitbucket.ioplus.clblas;

/**
 * Created by przemek on 24/08/15.
 */
public class Level2 {



    /**
     * Matrix-vector product with a general rectangular matrix and float elements.
     * Matrix-vector products:

     * \( y \leftarrow \alpha A x + \beta y \)
     * \( y \leftarrow \alpha A^T x + \beta y \)
     *
     * @param order BlasOrder.ROW_MAJOR|COLUMN_MAJOR
     * @param transA How matrix A is to be transposed: BlasTrans.NO_TRANS|TRANS|CONJ_TRANS.
     * @param M Number of rows in matrix A.
     * @param N Number of columns in matrix A.
     * @param alpha The factor of matrix A.
     * @param A Buffer object storing matrix A.
     * @param lda Leading dimension of matrix A. It cannot be less than N if ROW_MAJOR.
     * @param x Buffer object storing vector x.
     * @param beta The factor of the vector y.
     * @param y Buffer object storing the vector y.
     **/
    public static int sgemv(BlasOrder order,
                            BlasTrans transA,
                             int M,
                             int N,
                             float alpha,
                             final float[] A,
                             int lda,
                             final float[] x,
                             float beta,
                             float[] y){
        return BLAS.sgemv(
                order.ordinal(),
                transA.ordinal(),
                M, N,
                alpha, A, 0, lda, x, 0, 1, beta, y, 0, 1);
    }



    /**
     * vector-vector product with double elements and performs the rank 1 operation A <br>
     * Vector-vector products: <br>

     *   \( A \leftarrow \alpha X Y^T + A \)
     *
     * @param order BlasOrder.ROW_MAJOR|COLUMN_MAJOR
     * @param M Number of rows in matrix A.
     * @param N Number of columns in matrix A.
     * @param alpha specifies the scalar alpha.
     * @param X Buffer object storing vector X.
     * @param Y Buffer object storing vector Y.
     * @param A Buffer object storing matrix A. On exit, A is overwritten by the updated matrix.
     * @param lda Leading dimension of matrix A. It cannot be less than N.
     **/
    public static int sger(BlasOrder order,
                           int M,
                            int N,
                            float alpha,
                            final float[] X,
                            final float[] Y,
                            float[] A,
                            int lda){
        return BLAS.sger(BlasOrder.ROW_MAJOR.ordinal(),
                M, N, alpha, X, 0, 1, Y, 0, 1, A, 0, lda);
    }

}

package org.bitbucket.ioplus.clblas;

import org.bitbucket.ioplus.nativeloader.NativeLoader;

/**
 * Created by przemek on 28/08/15.
 */
public class BLAS {



    static {
        try {
            Class.forName("org.bitbucket.ioplus.clblas.clBLAS");
            NativeLoader.load(BLAS.class);
            init();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * setup
     */
    private static native int setup();

    /**
     * jvm hook on exit
     */
    private static native void tearDown();

    private static void init(){
        setup();
        Runtime.getRuntime().addShutdownHook(new Thread() {
            @Override
            public void run() {
                tearDown();
                System.out.println("tearDown()");
            }
        });
    }

    private BLAS(){}

    /** ***********************************************************************
     *
     *
     *
     *                 LEVEL 1 SUBROUTINES
     *
     *
     *
     *
     * ***********************************************************************/

    /**
     * Scales a float vector by a float constant.
     *
     * \( X \leftarrow \alpha X \)
     *
     * @param N    Number of elements in vector X
     * @param alpha The constant factor for vector X.
     * @param X    Buffer object storing vector X
     * @param offx Offset of first element of vector X in buffer object. Counted in elements.
     * @param incx Increment for the elements of X. Must not be zero.
     **/
    protected static native void sscal(int 	N,
                                    float 	alpha,
                                    float[] 	X,
                                    int 	offx,
                                    int 	incx);

    /**
     * Scale vector X of complex-float elements and add to Y.
     *
     *  \( Y \leftarrow \alpha X + Y \)
     *
     * @param N    Number of elements in vector X
     * @param alpha The constant factor for vector X.
     * @param X    Buffer object storing vector X
     * @param offx Offset of first element of vector X in buffer object. Counted in elements.
     * @param incx Increment for the elements of X. Must not be zero.
     * @param Y    Buffer object storing the vector Y.
     * @param offy Offset of first element of vector Y in buffer object. Counted in elements.
     * @param incy Increment for the elements of Y. Must not be zero.
     **/
    public static native void saxpy(int N,
                                    float alpha,
                                    final float[] X,
                                    int offx,
                                    int incx,
                                    float[] Y,
                                    int offy,
                                    int incy);


    /**
     * dot product of two vectors containing float-complex elements conjugating the first vector.
     *
     * @param N    Number of elements in vector X.
     * @param dotProduct Buffer object that will contain the dot-product value
     * @param offDP Offset to dot-product in dotProduct buffer object. Counted in elements.
     * @param X Buffer object storing vector X.
     * @param offx Offset of first element of vector X in buffer object. Counted in elements.
     * @param incx Increment for the elements of X. Must not be zero.
     * @param Y Buffer object storing the vector Y.
     * @param offy Offset of first element of vector Y in buffer object. Counted in elements.
     * @param incy Increment for the elements of Y. Must not be zero.
     **/
    public static native void sdot(int 	N,
                                   float[] 	dotProduct,
                                   int 	offDP,
                                   final float[] X,
                                   int 	offx,
                                   int 	incx,
                                   final float[] Y,
                                   int 	offy,
                                   int 	incy);

    /**
     * Copies float elements from vector X to vector Y.
     *
     * @param N    Number of elements in vector X
     * @param X    Buffer object storing vector X
     * @param offx Offset of first element of vector X in buffer object. Counted in elements.
     * @param incx Increment for the elements of X. Must not be zero.
     * @param Y    Buffer object storing the vector Y.
     * @param offy Offset of first element of vector Y in buffer object. Counted in elements.
     * @param incy Increment for the elements of Y. Must not be zero.
     **/
    protected static native void scopy(int N,
                                    final float[] X,
                                    int offx,
                                    int incx,
                                    float[] Y,
                                    int offy,
                                    int incy);


    /** ***********************************************************************
     *
     *
     *
     *                 LEVEL 2 SUBROUTINES
     *
     *
     *
     *
     * ***********************************************************************/


    /**
     * Matrix-vector product with a general rectangular matrix and float elements. Extended version.
     * Matrix-vector products:

     * \( y \leftarrow \alpha A x + \beta y \)
     * \( y \leftarrow \alpha A^T x + \beta y \)
     * @param blasOrder
     * @param transA How matrix A is to be transposed.
     * @param M Number of rows in matrix A.
     * @param N Number of columns in matrix A.
     * @param alpha The factor of matrix A.
     * @param A Buffer object storing matrix A.
     * @param offA Offset of the first element of the matrix A in the buffer object. Counted in elements.
     * @param lda Leading dimension of matrix A. It cannot be less than N.
     * @param x Buffer object storing vector x.
     * @param offx Offset of first element of vector x in buffer object. Counted in elements.
     * @param incx Increment for the elements of x. It cannot be zero.
     * @param beta The factor of the vector y.
     * @param y Buffer object storing the vector y.
     * @param offy Offset of first element of vector y in buffer object. Counted in elements.
     * @param incy Increment for the elements of y. It cannot be zero.
     **/
    public static native int sgemv(int blasOrder,
                                   int transA,
                                   int M,
                                   int N,
                                   float alpha,
                                   final float[] A,
                                   int offA,
                                   int lda,
                                   final float[] x,
                                   int offx,
                                   int incx,
                                   float beta,
                                   float[] y,
                                   int offy,
                                   int incy);




    /**
     * vector-vector product with double elements and performs the rank 1 operation A
     * Vector-vector products:

     * \( A \leftarrow \alpha X Y^T + A \)
     *
     * @param blasOrder
     * @param M Number of rows in matrix A.
     * @param N Number of columns in matrix A.
     * @param alpha specifies the scalar alpha.
     * @param X Buffer object storing vector X.
     * @param offx Offset in number of elements for the first element in vector X.
     * @param incx Increment for the elements of X. Must not be zero.
     * @param Y Buffer object storing vector Y.
     * @param offy Offset in number of elements for the first element in vector Y.
     * @param incy Increment for the elements of Y. Must not be zero.
     * @param A Buffer object storing matrix A. On exit, A is overwritten by the updated matrix.
     * @param offa Offset in number of elements for the first element in matrix A.
     * @param lda Leading dimension of matrix A. It cannot be less than N.
     **/
    public static native int sger(int blasOrder,
                                  int M,
                                  int N,
                                  float alpha,
                                  final float[] X,
                                  int offx,
                                  int incx,
                                  final float[] Y,
                                  int offy,
                                  int incy,
                                  float[] A,
                                  int offa,
                                  int lda);




    /** ***********************************************************************
     *
     *
     *
     *                 LEVEL 3 SUBROUTINES
     *
     *
     *
     *
     * ***********************************************************************/



    /**
     * Matrix-matrix product of general rectangular matrices with float elements. Extended version.
     * \( C \leftarrow \alpha A B + \beta C \)
     * \( C \leftarrow \alpha A^T B + \beta C \)
     * \( C \leftarrow \alpha A B^T + \beta C \)
     * \( C \leftarrow \alpha A^T B^T + \beta C \)
     *
     * @param blasOrder 0|1: ROW_MAJOR|COLUMN_MAJOR
     * @param transA How matrix A is to be transposed.
     * @param transB How matrix B is to be transposed.   Buffer object storing vector X
     * @param M      Number of rows in matrix A.
     * @param N      Number of columns in matrix B.
     * @param K      Number of columns in matrix A and rows in matrix B.
     * @param alpha  The factor of matrix A.
     * @param A      Buffer object storing matrix A.
     * @param offA   Offset of the first element of the matrix A in the buffer object. Counted in elements.
     * @param lda    Leading dimension of matrix A. It cannot be less than K.
     * @param B      Buffer object storing matrix B.
     * @param offB   Offset of the first element of the matrix B in the buffer object. Counted in elements.
     * @param ldb    Leading dimension of matrix B. It cannot be less than N.
     * @param beta   The factor of matrix C.
     * @param C      Buffer object storing matrix C.
     * @param offC   Offset of the first element of the matrix C in the buffer object. Counted in elements.
     * @param ldc    Leading dimension of matrix C. It cannot be less than N.
     **/
    public static synchronized native int sgemm(
            int blasOrder,
            int transA,
            int transB,
            int M,
            int N,
            int K,
            float alpha,
            final float[] A,
            int offA,
            int lda,
            final float[] B,
            int offB,
            int ldb,
            float beta,
            float[] C,
            int offC,
            int ldc);

}

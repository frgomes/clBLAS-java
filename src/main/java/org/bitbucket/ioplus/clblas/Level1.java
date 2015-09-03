package org.bitbucket.ioplus.clblas;

/**
 * Created by przemek on 29/08/15.
 */
public class Level1 {

    private Level1(){}


    /**
     * Scales a float vector by a float constant.
     *
     * \( X \leftarrow \alpha X \)
     *
     * @param N    Number of elements in vector X
     * @param alpha The constant factor for vector X.
     * @param X    Buffer object storing vector X
     **/
    public static void sscal(int N, float alpha, float[] X){

        BLAS.sscal(N, alpha, X, 0, 1);
    }
    /**
     * Scale vector X of complex-float elements and add to Y.
     *
     *  \( Y \leftarrow \alpha X + Y \)
     *
     * @param N    Number of elements in vector X
     * @param alpha The constant factor for vector X.
     * @param X    Buffer object storing vector X
     * @param Y    Buffer object storing the vector Y.
     **/
    public static void saxpy(int N, float alpha, final float[] X, float[] Y){
        BLAS.saxpy(N, alpha, X, 0, 1, Y, 0, 1);
    }
    /**
     * dot product of two vectors containing float-complex elements conjugating the first vector.
     *
     * @param N    Number of elements in vector X.
     * @param dotProduct Buffer object that will contain the dot-product value
     * @param X Buffer object storing vector X.
     * @param Y Buffer object storing the vector Y.
     **/
    public static void sdot(int N, float[] dotProduct, final float[] X, final float[] Y){
        BLAS.sdot(N, dotProduct, 0, X, 0, 1, Y, 0, 1);
    }

    /**
     * Copies float elements from vector X to vector Y.
     *
     * @param N    Number of elements in vector X
     * @param X    Buffer object storing vector X
     * @param Y    Buffer object storing the vector Y.
     **/
    public static void scopy(int N, final  float[] X, float[] Y){
        BLAS.scopy(N, X, 0, 1, Y, 0, 1);
    }



}

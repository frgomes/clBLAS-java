package org.bitbucket.ioplus.math;

import org.bitbucket.ioplus.clblas.BlasOrder;
import org.bitbucket.ioplus.clblas.BlasTrans;
import org.bitbucket.ioplus.clblas.Level1;
import org.bitbucket.ioplus.clblas.Level3;

import java.util.Random;

import  static org.bitbucket.ioplus.clblas.Level3.*;

/**
 * Created by przemek on 13/08/15.
 */
public class Algebra {

    public static float[] transpose(final float[] A, int lda) throws Exception {
        int err = 0;
        float[] C = new float[A.length];
        int ldc = A.length/lda;
        int M = lda, N = ldc, K = ldc;
        float alpha = 1f, beta = 1f;
        float[] I = identity(ldc);
        err = sgemm(
                BlasOrder.ROW_MAJOR,
                BlasTrans.TRANS, BlasTrans.NO_TRANS,
                M,N,K,
                alpha,A,lda,
                I, ldc,
                beta,C,ldc);
        if(err!=0){
            throw new Exception("sgemm subroutines fails.");
        }
        return C;
    }

    public static float[] identity(int lda){
        float[] I = new float[lda*lda];
        for(int i = 0; i < lda; i++){
            int idx = i+(i*lda);
            I[idx] = 1;
        }
        return I;
    }

    public  static float[] add(final float[] a, final float[] b){
        Level1.saxpy(a.length, 1f, a, b);
        return b;
    }

    /**
     *
     * Matrix Ops
     *
     */

    public  static Matrix transpose(final Matrix M){
        if(M.n == 1 || M.m == 1){
            return new Matrix(M.getData(), M.n, M.m);
        } else {
            try {
                return new Matrix(transpose(M.getData(), M.ld), M.n, M.m);
            } catch (Exception e) {
                e.printStackTrace();
                return M;
            }
        }
    }

    /**
     *
     * @param A
     * @param B
     * @return
     */
    public  static Matrix add(final Matrix A, final Matrix B){
        try {
            if(A.m==B.m && A.n==B.n) {
                throw new Matrix.WrongDimensionsException();
            }
        } catch (Matrix.WrongDimensionsException e){
        }
        float[] cData = add(A.getData(), B.getData());
        return new Matrix(cData, A.m, A.n);

    }

    /**
     *
     * @param A
     * @param B
     * @return
     */
    public static Matrix mmul(final Matrix A, final Matrix B){
        try{
            if(A.n!=B.m) throw new Matrix.WrongDimensionsException();
        } catch (Matrix.WrongDimensionsException e) {
            e.printStackTrace();
            System.exit(1);
        }
        float[] result = new float[A.m * B.n];
        Level3.sgemm(BlasOrder.ROW_MAJOR,
                BlasTrans.NO_TRANS,
                BlasTrans.NO_TRANS,
                A.m, B.n, A.n,
                1f, A.getData(), A.ld,
                B.getData(), B.ld,
                1f, result, B.ld);
        return new Matrix(result, A.m, B.n);
    }



    /**
     *
     * @param A
     * @param B
     * @return
     */
    public  static float dot(Matrix A, Matrix B){
        int aLen = A.m*A.n;
        int bLen = B.m*B.n;
        try{
            if(bLen!=aLen) throw new Matrix.WrongDimensionsException();
        } catch (Matrix.WrongDimensionsException e) {
            e.printStackTrace();
            System.exit(1);
        }
        float[] dotP = new float[1];

        Level1.sdot(aLen, dotP, A.getData(), B.getData());
        return dotP[0];
    }


    /**
     *
     * @param M
     * @param alpha
     * @return
     */
    public  static Matrix scale(Matrix M, float alpha){
        float[] scaled = M.getData().clone();
        Level1.sscal(scaled.length, alpha, scaled);
        return new Matrix(scaled, M.m, M.n);
    }

    /**
     *
     * @param A
     * @param B
     * @return
     */
    public  static Matrix vstack(Matrix A, Matrix B){
        int aLen = A.m*A.n;
        int bLen = B.m*B.n;
        float[] C = new float[aLen+bLen];
        System.arraycopy(A.getData(),0, C,0, aLen);
        System.arraycopy(B.getData(), 0, C, aLen, bLen);
        return new Matrix(C, A.m+B.m, Math.max(A.n, B.n));
    }

    /**
     *
     * @param A
     * @param B
     * @return
     */
    public static Matrix hstack(Matrix A, Matrix B){
        float[] C = new float[A.getData().length+B.getData().length];
        return transpose(vstack(transpose(A),transpose(B)));
    }

    /**
     *
     * @param M
     * @return
     */
    public  static Matrix tanh(Matrix M){
        float[] a = M.getData();
        for(int i = 0; i < a.length;i++){
            a[i]= (float)Math.tanh(a[i]);
        }
        return new Matrix(a, M.m, M.n);
    }

    /**
     *
     * @param m
     * @return
     */
    public  static Matrix eye(int m){
        return new Matrix(identity(m), m,m);
    }

    /**
     *
     * @param m
     * @param n
     * @param eta
     * @return
     */
    public  static Matrix rand(int m, int n, float eta){
        Random generator = new Random(42);
        float[] R = new float[m*n];
        for(int i = 0; i < R.length; i++){
            R[i] = (generator.nextFloat()-0.5f)*eta;
        }
        return new Matrix(R, m,n);
    }

    /**
     *
     * @param m
     * @param n
     * @param eta
     * @return
     */
    public static Matrix fill(int m, int n, float eta){
        float[] R = new float[m*n];
        for(int i = 0; i < R.length; i++){
            R[i] = eta;
        }
        return new Matrix(R, m,n);
    }

    /**
     *
     * @param m
     */
    public static void size(Matrix m){
        System.out.printf("[%d, %d]\n", m.m, m.n);
    }

    /**
     *
     * @param m
     */
    public static void print(Matrix m){
        size(m);
        if(m.m*m.n>50*50){
            return;
        }
        for(int row = 0; row < m.m; row++){
            for(int col = 0; col < m.ld; col++){
                float v = m.get(row,col);
                System.out.printf(" %.3f, ", v);
            }
            System.out.println();
        }
    }


}

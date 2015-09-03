package org.bitbucket.ioplus.math;

import org.jblas.FloatMatrix;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.util.Random;

import static junit.framework.Assert.assertEquals;
import static junit.framework.Assert.assertFalse;
import static org.junit.Assert.assertArrayEquals;

/**
 * Created by przemek on 19/08/15.
 */
public class AlgebraTests {
    private Matrix A, Asmall;
    int T = 10;

    @Before
    public void setupMatrices(){
        A = Algebra.rand(1000,1000, 1f);
        float[] aSmallData = new float[10*5];
        for(int i = 0; i < aSmallData.length;i++){
            aSmallData[i]=i+1;
        }
        Asmall = new Matrix(aSmallData,10,5);
    }



    @Test
    public void identityTest(){
        int lda = 20;
        float[] I = Algebra.identity(lda);
        printMatrix(I, lda);
    }

    @Test
    public void transposeTest() throws Exception{
        float[] a = {
                1,2,3,4,
                5,6,7,8
        };
        int lda = 4;
        float[] t = Algebra.transpose(a,lda);
        printMatrix(t,2);

        lda = 1000;
        int ldc = 4000;
        a = new float[ldc*lda];
        Random r = new Random();
        for(int i = 0; i < a.length; i++) a[i]=r.nextFloat();

        int T = 1;
        long start = System.currentTimeMillis();
        t = Algebra.transpose(a,lda);
        long end = System.currentTimeMillis();
        System.out.printf("Elapsed time is %.3f seconds.\n", ((end - start) / T) * 0.001);
    }

    @Test
    public void addTest(){
        float[] a = new float[10];
        for(int i = 0; i < 10; i++) a[i] = i;
        float[] c = Algebra.add(a, a);
        printMatrix(c, c.length);

    }

    @Test
    public void matrixTranspose(){
        Algebra.print(Asmall);
        for(int t = 0; t < T; t++) {
            Algebra.print(Algebra.transpose(Asmall));
        }
    }

    @Test
    public void matrixTEst(){
        float[] a = new float[25];
        for(int i = 0; i < 25; i++) a[i] = i;
        Matrix A = new Matrix(a, 5,5);
        Matrix B = new Matrix(Algebra.identity(5), 5,5);
        Matrix C = Algebra.mmul(A, B);
       Algebra.print(C);

        Matrix I = new Matrix(Algebra.eye(4000));
        System.out.println("__________");
        long start = System.currentTimeMillis();
        Algebra.mmul(I,I);
        long end = System.currentTimeMillis();
        System.out.printf("Elapsed time is %.3f seconds.\n", ((end - start) / 1) * 0.001);


    }

    @Test public void dotTestM(){
        int m = 10000;
        float[] a = new float[m];
        for(int i = 0; i < m; i++) a[i] = i;
        Matrix A = new Matrix(a, 1,m);
        float d = Algebra.dot(A, A);
        System.out.println(d);
    }
    @Test public void  hstackTest(){
        float[] a = new float[25];
        for(int i = 0; i < 25; i++) a[i] = i;
        Matrix A = new Matrix(a, 25,1);
        Matrix B = Algebra.hstack(A, A);
        printMatrix(B.getData(), 2);
    }

    @Test public void vstackTest(){
        Matrix e = new Matrix(new float[]{}, 0,0);
        int n = 10;
        float[] a = new float[n];
        for(int i = 0; i < n; i++) a[i] = i;
        Matrix A = new Matrix(a, 10,1);
        Matrix B = Algebra.vstack(e, A);
        //System.out.print(B.ld);
        Algebra.print(B);

//        for(int i = 0; i < B.getData().length; i++){
//            System.out.println(B.getData()[i]);
//        }
    }

    @Test public void scaleTest(){
        int n = 10;
        float[] a = new float[n];
        for(int i = 0; i < n; i++) a[i] = i;
        Matrix A = new Matrix(a, 5,2);
        Matrix B = Algebra.scale(A,0.1f);
        Algebra.print(B);

    }
    @Test public void immutabilityTest(){
        int n = 10;
        float[] a = new float[n];
        for(int i = 0; i < n; i++) a[i] = i;
        Matrix A = new Matrix(a, 10,1);
        Matrix B = Algebra.scale(A,0.1f);

        A.getData()[0] = 10;

        Assert.assertTrue(A.getData()[0] == a[0]);
    }



    @Test
    public void matrixConstructorTest(){
        float[] a = new float[]{1,2,3,4,5};
        Matrix A = new Matrix(a,5,1);
        Matrix B = new Matrix(a,1,5);
        Algebra.size(A);
        Algebra.size(B);
        a[0] = 6;
        assertFalse(a[0] == A.getData()[0]);
        assertFalse(a[0] == B.getData()[0]);
    }

    @Test
    public void exceptionTest(){
        new Matrix(new float[4], 4, 4);
    }

    @Test
    public void print(){
        Algebra.print(Algebra.rand(1, 10, 1f));
    }

    @Test
    public void transpose2(){
        int m = 1000, n = 1000;
        Matrix a = Algebra.rand(m,n, 1f);
        Matrix b = Algebra.transpose(a);

        float[][] data = new float[m][n];
        float[] aData = a.getData();
        for(int row = 0 ; row < m; row++){
            for(int col = 0; col <n; col++){
                data[row][col] = aData[col+row*n];
            }
        }

        FloatMatrix floatMatrix = new FloatMatrix(data);
//        System.out.println(floatMatrix.columns);
//        floatMatrix.print();
//        floatMatrix.getRow(0).print();
//        Algebra.print(a);
//        for(int i = 0; i<floatMatrix.data.length; i++){
//            System.out.printf("%.3f ", floatMatrix.data[i]);
//        }
//        System.out.println();


        assertArrayEquals(b.getData(), floatMatrix.data, 1e-4f);
        assertArrayEquals(Algebra.transpose(b).getData(), floatMatrix.transpose().data, 1e-4f);



    }

    @Test
    public void mmul(){
        Matrix a = new Matrix(new float[]
                {1f,2f,3f,4f,5f,6f,7f,8f,9f}, 3,3);
        Algebra.print(a);
        Matrix b = new Matrix(new float[]
                {1,2,3}, 3,1);
        Algebra.print(b);
        Algebra.print(Algebra.mmul(a,b));
    }


    private void printMatrix(float[] A, int lda){
        for(int i = 0; i< A.length; i++){
            System.out.printf("%.3f\t", A[i]);
            if((i+1)%lda==0 && i!=0) System.out.println();
        }
    }





}

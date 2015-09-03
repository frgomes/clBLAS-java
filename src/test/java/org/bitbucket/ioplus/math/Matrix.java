package org.bitbucket.ioplus.math;


public class Matrix {
    /**
     * Read-only field.
     */
    private final float[] xs;
    public float[] getData(){
        return xs.clone();
    }

    /**
     *
     * @param row
     * @param col
     * @return x
     */
    public float get(int row, int col){
        return xs[col + row*ld];
    }

    /**
     * Read-only fields.
     */
    public final int m, n, ld;

    /**
     * Empty matrix. [0,0] dimension.
     */
    public Matrix(){
        xs = new float[0];
        n = 0; m = 0; ld = 0;
    }

    /**
     * Zero matrix [m, n] dimension.
     * @param m
     * @param n
     */
    public Matrix(int m, int n){
        xs = new float[m*n];
        this.n = n; this.m = m; this.ld = n;
    }

    /**
     * Simple, float precision matrix as immutable pojo.
     * Row major order.
     *
     * @param xs data
     * @param m
     * @param n
     */
    public Matrix(float[] xs, int m, int n){
        try {
            checkDimensions(xs.length,m,n);
        } catch (WrongDimensionsException e) {
            e.printStackTrace();
        }
        this.xs = xs.clone();
        this.n = n;
        this.m = m;
        this.ld = n;
    }


    /**
     * Matrix from Matrix.
     * @param M Matrix to be cloned.
     */
    public Matrix(Matrix M){
        xs = M.xs.clone();
        m = M.m;
        n = M.n;
        ld = n;
    }

    private void checkDimensions(int len, int m, int n) throws WrongDimensionsException {
        if (len!=m*n) throw new WrongDimensionsException();
    }

    public static class WrongDimensionsException extends Exception {
    }


}

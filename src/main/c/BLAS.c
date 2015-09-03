#include <sys/types.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <clBLAS.h>

#include <jni.h>
#include "org_bitbucket_ioplus_clblas_BLAS.h"


cl_platform_id platform;
cl_device_id  device;
cl_context_properties props[3];
cl_context ctx;
cl_command_queue queue;



/** ***********************************************************************
 *
 *
 *
 *                          SETUP AND TEAR DOWN
 *
 *
 *
 *
 * ***********************************************************************/




/*
 * Class:     org_bitbucket_ioplus_clblas_BLAS
 * Method:    tearDown
 * Signature: ()V
 */

JNIEXPORT void JNICALL Java_org_bitbucket_ioplus_clblas_BLAS_tearDown
(JNIEnv * env, jclass class){
    /* Finalize work with clblas. */
    clblasTeardown();

    /* Release OpenCL working objects. */
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
 }

/*
 * Class:     org_bitbucket_ioplus_clblas_BLAS
 * Method:    setup
 * Signature: ()I
 */

JNIEXPORT jint JNICALL Java_org_bitbucket_ioplus_clblas_BLAS_setup(
    JNIEnv * env, jclass class){
    static int is_setup = 0;

    if(is_setup!=0){
        return 0;
    }

    cl_uint major,minor,patch;
    clblasStatus status;

    status = clblasGetVersion(&major,&minor,&patch);
    if (status != CL_SUCCESS) {
        printf("clblasGetVersion() failed with %d\n", status);
        return 1;
    }
    printf("clblas version %d.%d.%d\n", major,minor,patch);


    props[0] = CL_CONTEXT_PLATFORM;

    cl_int err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) {
        printf( "clGetPlatformIDs() failed with %d\n", err );
        return err;
    }
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        printf( "GPU not available. Trying DEFAULT\n" );
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, NULL);
        if (err != CL_SUCCESS) {
            printf( "clGetDeviceIDs() failed with %d\n", err );
            return err;
        }
    }

    props[1] = (cl_context_properties)platform;

    ctx = clCreateContext(props, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        printf( "clCreateContext() failed with %d\n", err );
        return err;
    }

    queue = clCreateCommandQueue(ctx, device, 0, &err);
    if (err != CL_SUCCESS) {
        printf( "clCreateCommandQueue() failed with %d\n", err );
        clReleaseContext(ctx);
        return err;
    }

    /* Setup clblas. */
    err = clblasSetup();
    if (err != CL_SUCCESS) {
        printf("clblasSetup() failed with %d\n", err);
        clReleaseCommandQueue(queue);
        clReleaseContext(ctx);
        return err;
    }

    is_setup = 1;
    return err;
}


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

/*
 * Class:     org_bitbucket_ioplus_clblas_BLAS
 * Method:    sscal
 * Signature: (IF[FII)V
 */
JNIEXPORT void JNICALL Java_org_bitbucket_ioplus_clblas_BLAS_sscal
(JNIEnv * env, jclass class,
 jint N,
 jfloat alpha,
 jfloatArray x,
 jint offx,
 jint incx){

    cl_int err;

    cl_mem bufX;
    cl_event event = NULL;

    //int ret = 0;

    (*env)->MonitorEnter(env, x);


    cl_float *X = (*env)->GetPrimitiveArrayCritical(env, x, 0);

    int xArrayLength = (*env)->GetArrayLength(env,x);

    jint lenX = xArrayLength;


    int n = (xArrayLength-offx)/abs(incx);
    // lenY = N;
    if(N!=n){
        N=n;
    }

        /* Prepare OpenCL memory objects and place vectors inside them. */
    bufX = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (lenX * sizeof(cl_float)),
                          NULL, &err);

    err = clEnqueueWriteBuffer(queue, bufX, CL_TRUE, 0,
        ( lenX * sizeof(cl_float)), X, 0, NULL, NULL);

    /* Call clblas function. */
    err = clblasSscal(N, alpha, bufX,
                       offx, incx,
                       1, &queue, 0, NULL, &event);

    if (err != CL_SUCCESS) {
        printf("clblasSscal() failed with %d\n", err);
    }
    else {
        /* Wait for calculations to be finished. */
        err = clWaitForEvents(1, &event);

        /* Fetch results of calculations from GPU memory. */
        err = clEnqueueReadBuffer(queue, bufX, CL_TRUE, 0,
                                  (xArrayLength * sizeof(cl_float)),
                                  //(lenX * sizeof(cl_float)),
                                  X, 0, NULL, NULL);
    }

    /* Release OpenCL memory objects. */
    clReleaseMemObject(bufX);

    (*env)->ReleasePrimitiveArrayCritical(env, x, X, 0);
    (*env)->MonitorExit(env, x);
}

//////////////////////////////////////////////////////////////////////////////

/*
 * Class:     org_bitbucket_ioplus_clblas_BLAS
 * Method:    saxpy
 * Signature: (IF[FII[FII)V
 */
JNIEXPORT void JNICALL Java_org_bitbucket_ioplus_clblas_BLAS_saxpy
(JNIEnv * env, jclass class,
 jint N, jfloat alpha, jfloatArray x, jint offx, jint incx,
 jfloatArray y, jint offy, jint incy){
    cl_int err;

    cl_mem bufX, bufY;
    cl_event event = NULL;



    int xArrayLength = (*env)->GetArrayLength(env,x);
    int yArrayLength = (*env)->GetArrayLength(env,y);

    jint lenX = xArrayLength;
    jint lenY = yArrayLength;


    int n = (xArrayLength-offx)/abs(incx);
    // lenY = N;
    if(N!=n){
        N=n;
    }

    (*env)->MonitorEnter(env, x);
    (*env)->MonitorEnter(env, y);

    cl_float *X = (*env)->GetPrimitiveArrayCritical(env, x, 0);
    cl_float *Y = (*env)->GetPrimitiveArrayCritical(env, y, 0);

    /* Prepare OpenCL memory objects and place vectors inside them. */
    bufX = clCreateBuffer(ctx, CL_MEM_READ_WRITE, ( lenX * sizeof(cl_float)),
                          NULL, &err);

    err = clEnqueueWriteBuffer(queue, bufX, CL_TRUE, 0,
                               ( lenX * sizeof(cl_float)), X, 0, NULL, NULL);

    bufY = clCreateBuffer(ctx, CL_MEM_READ_WRITE, ( lenY * sizeof(cl_float)),
                          NULL, &err);

    err = clEnqueueWriteBuffer(queue, bufY, CL_TRUE, 0,
                               ( lenY * sizeof(cl_float)), Y, 0, NULL, NULL);

    /* Call clblas function. */
    err = clblasSaxpy( N, alpha, bufX, offx, incx, bufY, offy, incy, 1, &queue, 0, NULL, &event);
    if (err != CL_SUCCESS) {
        printf("clblasSaxpy() failed with %d\n", err);
    }
    else {
        /* Wait for calculations to be finished. */
        err = clWaitForEvents(1, &event);

        /* Fetch results of calculations from GPU memory. */
        err = clEnqueueReadBuffer(queue, bufY, CL_TRUE, 0,
                                  (lenY*sizeof(cl_float)),
                                  Y, 0, NULL, NULL);

    }

    /* Release OpenCL memory objects. */
    clReleaseMemObject(bufY);
    clReleaseMemObject(bufX);

    (*env)->ReleasePrimitiveArrayCritical(env, x, X, 0);
    (*env)->ReleasePrimitiveArrayCritical(env, y, Y, 0);
    (*env)->MonitorExit(env, x);
    (*env)->MonitorExit(env, y);
}

//////////////////////////////////////////////////////////////////////////

/*
 * Class:     org_bitbucket_ioplus_clblas_BLAS
 * Method:    sdot
 * Signature: (I[FI[FII[FII)V
 */
JNIEXPORT void JNICALL Java_org_bitbucket_ioplus_clblas_BLAS_sdot
(JNIEnv * env, jclass class,
 jint N,
 jfloatArray dot_product,
 jint offDP,
 jfloatArray x,
 jint offx,
 jint incx,
 jfloatArray y,
 jint offy,
 jint incy){

    cl_int err;

    cl_mem bufX, bufY, bufDotP, scratchBuff;
    cl_event event = NULL;
    /* int lenX = 1 + (N-1)*abs(incx); */
    /* int lenY = 1 + (N-1)*abs(incy); */


    int xArrayLength = (*env)->GetArrayLength(env,x);
    int yArrayLength = (*env)->GetArrayLength(env,y);

    (*env)->MonitorEnter(env, x);
    (*env)->MonitorEnter(env, y);
    (*env)->MonitorEnter(env, dot_product);

    jint lenX = xArrayLength;
    jint lenY = yArrayLength;


    int n = (xArrayLength-offx)/abs(incx);
    // lenY = N;
    if(N!=n){
        N=n;
    }

    cl_float *X = (*env)->GetPrimitiveArrayCritical(env, x, 0);
    cl_float *Y = (*env)->GetPrimitiveArrayCritical(env, y, 0);
    cl_float dotProduct;
    cl_float *dot = (*env)->GetPrimitiveArrayCritical(env,dot_product, 0);


    /* Prepare OpenCL memory objects and place matrices inside them. */
    bufX = clCreateBuffer(ctx, CL_MEM_READ_ONLY, (lenX*sizeof(cl_float)), NULL, &err);
    bufY = clCreateBuffer(ctx, CL_MEM_READ_ONLY, (lenY*sizeof(cl_float)), NULL, &err);
    // Allocate 1 element space for dotProduct
    bufDotP = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, (sizeof(cl_float)), NULL, &err);
    // Allocate minimum of N elements
    scratchBuff = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (N*sizeof(cl_float)), NULL, &err);

    err = clEnqueueWriteBuffer(queue, bufX, CL_TRUE, 0, (lenX*sizeof(cl_float)), X, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, bufY, CL_TRUE, 0, (lenY*sizeof(cl_float)), Y, 0, NULL, NULL);

    /* Call clblas function. */
    err = clblasSdot( N, bufDotP, offDP, bufX, offx, incx, bufY, offy, incy, scratchBuff,
                                    1, &queue, 0, NULL, &event);
    if (err != CL_SUCCESS) {
        printf("clblasSdot() failed with %d\n", err);
    }
    else {
        /* Wait for calculations to be finished. */
        err = clWaitForEvents(1, &event);

        /* Fetch results of calculations from GPU memory. */
        err = clEnqueueReadBuffer(queue, bufDotP, CL_TRUE, 0, sizeof(cl_float),
                                    &dotProduct, 0, NULL, NULL);
    }

    /* Release OpenCL memory objects. */
    clReleaseMemObject(bufY);
    clReleaseMemObject(bufX);
    clReleaseMemObject(bufDotP);
    clReleaseMemObject(scratchBuff);



    dot[0] = dotProduct;
    (*env)->ReleasePrimitiveArrayCritical(env, x, X, 0);
    (*env)->ReleasePrimitiveArrayCritical(env, y, Y, 0);
    (*env)->ReleasePrimitiveArrayCritical(env, dot_product, dot, 0);


    (*env)->MonitorExit(env, x);
    (*env)->MonitorExit(env, y);
    (*env)->MonitorExit(env, dot_product);


}

/*
 * Class:     org_bitbucket_ioplus_clblas_BLAS
 * Method:    scopy
 * Signature: (I[FII[FII)V
 */
JNIEXPORT void JNICALL Java_org_bitbucket_ioplus_clblas_BLAS_scopy
(JNIEnv * env, jclass class,
 jint N,
 jfloatArray x,
 jint offx,
 jint incx,
 jfloatArray y,
 jint offy,
 jint incy){

    cl_int err;

    cl_mem bufX, bufY;
    cl_event event = NULL;


    cl_float *X = (*env)->GetPrimitiveArrayCritical(env, x, 0);
    // X = X + offx;

    cl_float *Y = (*env)->GetPrimitiveArrayCritical(env, y, 0);
    //Y = Y + offy;

    int xArrayLength = (*env)->GetArrayLength(env,x);
    int yArrayLength = (*env)->GetArrayLength(env,y);

    /*
    int lenX =  1 + (xArrayLength-1)*abs(incx);
    int lenY = 1 + (yArrayLength-1)*abs(incy);
    */

    int lenX = xArrayLength;// 1 + (N-1)*abs(incx);
    //lenX = N;
    int lenY = yArrayLength;//1 + (N-1)*abs(incy);

    int n = (xArrayLength-offx)/abs(incx);
    // lenY = N;
    if(N!=n){
        N=n;
    }


    /* Prepare OpenCL memory objects and place vectors inside them. */
    bufX = clCreateBuffer(ctx, CL_MEM_READ_ONLY, (lenX * sizeof(cl_float)),
                          NULL, &err);

    err = clEnqueueWriteBuffer(queue, bufX, CL_TRUE, 0,
                               ( lenX * sizeof(cl_float)), X, 0, NULL, NULL);

    bufY = clCreateBuffer(ctx, CL_MEM_READ_WRITE, ( lenY * sizeof(cl_float)),
                          NULL, &err);

    err = clEnqueueWriteBuffer(queue, bufY, CL_TRUE, 0,
                               ( lenY * sizeof(cl_float)), Y, 0, NULL, NULL);

    /* Call clblas function. */
    err = clblasScopy(N, bufX, offx, incx, bufY, offy, incy, 1, &queue, 0, NULL, &event);
    if (err != CL_SUCCESS) {
        printf("clblasScopy() failed with %d\n", err);
    }
    else {
        /* Wait for calculations to be finished. */
        err = clWaitForEvents(1, &event);

        /* Fetch results of calculations from GPU memory. */
        err = clEnqueueReadBuffer(queue, bufY, CL_TRUE,
                                  0,
                                  (lenY * sizeof(cl_float)),
                                  Y, 0, NULL, NULL);

    }

    /* Release OpenCL memory objects. */
    clReleaseMemObject(bufY);
    clReleaseMemObject(bufX);


    (*env)->ReleasePrimitiveArrayCritical(env, x, X, 0);
    (*env)->ReleasePrimitiveArrayCritical(env, y, Y, 0);


}















/* ***********************************************************************
 *
 *
 *
 *                 LEVEL 2 SUBROUTINES
 *
 *
 *
 *
 * ***********************************************************************/





/*
 * Class:     org_bitbucket_ioplus_clblas_BLAS
 * Method:    sgemv
 * Signature: (IIIIF[FII[FIIF[FII)I
 */
JNIEXPORT jint JNICALL Java_org_bitbucket_ioplus_clblas_BLAS_sgemv
(JNIEnv * env, jclass class,
 jint order,
 jint transA,
 jint M,
 jint N,
 jfloat alpha,
 jfloatArray a,
 jint offA,
 jint lda,
 jfloatArray x,
 jint offX,
 jint incx,
 jfloat beta,
 jfloatArray y,
 jint offY,
 jint incy){

    cl_int err;

    cl_mem bufA, bufX, bufY;
    cl_event event = NULL;

    /* jni arguments */

    cl_float * A;
    A = (*env)->GetPrimitiveArrayCritical(env, a, 0);
    cl_float * X;
    X = (*env)->GetPrimitiveArrayCritical(env, x, 0);
    cl_float * Y;
    Y = (*env)->GetPrimitiveArrayCritical(env, y, 0);

    int A_len = (*env)->GetArrayLength(env,a);
    int X_len = (*env)->GetArrayLength(env,x);
    int Y_len = (*env)->GetArrayLength(env,y);

/* Prepare OpenCL memory objects and place matrices inside them. */
    bufA = clCreateBuffer(ctx, CL_MEM_READ_ONLY, A_len * sizeof(*A),
                          NULL, &err);
    bufX = clCreateBuffer(ctx, CL_MEM_READ_ONLY, X_len * sizeof(*X),
                          NULL, &err);
    bufY = clCreateBuffer(ctx, CL_MEM_READ_WRITE, Y_len * sizeof(*Y),
                          NULL, &err);

    err = clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0,
        A_len * sizeof(*A), A, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, bufX, CL_TRUE, 0,
        X_len * sizeof(*X), X, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, bufY, CL_TRUE, 0,
        Y_len * sizeof(*Y), Y, 0, NULL, NULL);

    /* Call clblas extended function. */
    err = clblasSgemv((clblasOrder)order, transA, M, N, alpha,
                           bufA, offA, lda, bufX, offX, incx, beta,
                           bufY, offY, incy, 1, &queue, 0, NULL, &event);
    if (err != CL_SUCCESS) {
        printf("clblasSgemvEx() failed with %d\n", err);
        return 1;
    }
    else {
        /* Wait for calculations to be finished. */
        err = clWaitForEvents(1, &event);

        /* Fetch results of calculations from GPU memory. */
        err = clEnqueueReadBuffer(queue, bufY, CL_TRUE, 0, Y_len * sizeof(*Y),
                                  Y, 0, NULL, NULL);
    }

    /* Release OpenCL memory objects. */
    clReleaseMemObject(bufY);
    clReleaseMemObject(bufX);
    clReleaseMemObject(bufA);

    /* Finalize work with clblas. */
    //clblasTeardown();

    /* Release OpenCL working objects. */
    //clReleaseCommandQueue(queue);
    //clReleaseContext(ctx);


    (*env)->ReleasePrimitiveArrayCritical(env, a, A, 0);
    (*env)->ReleasePrimitiveArrayCritical(env, x, X, 0);
    (*env)->ReleasePrimitiveArrayCritical(env, y, Y, 0);

    return err;
}
/*
 * Class:     org_bitbucket_ioplus_clblas_BLAS
 * Method:    sger
 * Signature: (IIIF[FII[FII[FII)I
 */
JNIEXPORT jint JNICALL Java_org_bitbucket_ioplus_clblas_BLAS_sger
(JNIEnv *env, jclass class,
 jint order,
 jint M,
 jint N,
 jfloat alpha,
 jfloatArray x,
 jint offx,
 jint incx,
 jfloatArray y,
 jint offy,
 jint incy,
 jfloatArray a,
 jint offa,
 jint lda){

    cl_int err;

    cl_mem bufX, bufY, bufA;
    cl_event event = NULL;

    err = 0;

    /* jni arguments */

    cl_float * A;
    A = (*env)->GetPrimitiveArrayCritical(env, a, 0);
    cl_float * X;
    X = (*env)->GetPrimitiveArrayCritical(env, x, 0);
    cl_float * Y;
    Y = (*env)->GetPrimitiveArrayCritical(env, y, 0);

    int A_len = (*env)->GetArrayLength(env,a);
    int X_len = (*env)->GetArrayLength(env,x);
    int Y_len = (*env)->GetArrayLength(env,y);

    X_len = 1 + (X_len - 1) * abs(incx);
    Y_len = 1 + (Y_len - 1) * abs(incy);

    printf("X_len =  %d\n", X_len);
    printf("Y_len =  %d\n", Y_len);

    /* Prepare OpenCL memory objects and place matrices inside them. */
    /* bufA = clCreateBuffer(ctx, CL_MEM_READ_WRITE, M * lda * sizeof(cl_float), */
    /*                       NULL, &err); */
    /* bufX = clCreateBuffer(ctx, CL_MEM_READ_ONLY, ( 1 + ( M - 1 )*abs( incx ) ) * sizeof(cl_float), */
    /*                       NULL, &err); */
    /* bufY = clCreateBuffer(ctx, CL_MEM_READ_ONLY, ( 1 + ( N - 1 )*abs( incy ) ) * sizeof(cl_float), */
    /*                       NULL, &err); */
    bufA = clCreateBuffer(ctx, CL_MEM_READ_ONLY, A_len * sizeof(*A),
                          NULL, &err);
    bufX = clCreateBuffer(ctx, CL_MEM_READ_ONLY, X_len * sizeof(*X),
                          NULL, &err);
    bufY = clCreateBuffer(ctx, CL_MEM_READ_WRITE, Y_len * sizeof(*Y),
                          NULL, &err);


    /* err = clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, */
    /*                            M * lda * sizeof(cl_float), A, 0, NULL, NULL); */
    /* err = clEnqueueWriteBuffer(queue, bufX, CL_TRUE, 0, */
    /*                            ( 1 + ( M - 1 )*abs( incx ) ) * sizeof(cl_float), X, 0, NULL, NULL); */
    /* err = clEnqueueWriteBuffer(queue, bufY, CL_TRUE, 0, */
    /*                            ( 1 + ( N - 1 )*abs( incy ) ) * sizeof(cl_float), Y, 0, NULL, NULL); */


    err = clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0,
                               A_len * sizeof(*A), A, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, bufX, CL_TRUE, 0,
                               X_len * sizeof(*X), X, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, bufY, CL_TRUE, 0,
                               Y_len * sizeof(*Y), Y, 0, NULL, NULL);


    /* Call clblas function. */
    err = clblasSger((clblasOrder)order,
                     M, N,
                     alpha, bufX, offx, incx,
                     bufY, offy, incy,
                     bufA, offa, lda,
                     1, &queue, 0, NULL, &event);
    if (err != CL_SUCCESS) {
        printf("clblasSger() failed with %d\n", err);
        return 1;
    }
    else {
        /* Wait for calculations to be finished. */
        err = clWaitForEvents(1, &event);

        /* Fetch results of calculations from GPU memory. */
     /*    err = clEnqueueReadBuffer(queue, bufA, CL_TRUE, 0, (M * lda * sizeof(cl_float)), */
    /*                               A, 0, NULL, NULL); */
    /* } */
    err = clEnqueueReadBuffer(queue, bufA, CL_TRUE, 0, A_len* sizeof(cl_float),
                              A, 0, NULL, NULL);
    }

    /* Release OpenCL memory objects. */
    clReleaseMemObject(bufY);
    clReleaseMemObject(bufX);
    clReleaseMemObject(bufA);


    (*env)->ReleasePrimitiveArrayCritical(env, a, A, 0);
    (*env)->ReleasePrimitiveArrayCritical(env, x, X, 0);
    (*env)->ReleasePrimitiveArrayCritical(env, y, Y, 0);

    return err;

}


/* ***********************************************************************
 *
 *
 *
 *                 LEVEL 3 SUBROUTINES
 *
 *
 *
 *
 * ***********************************************************************/


/*
 * Class:     org_bitbucket_ioplus_clblas_BLAS
 * Method:    sgemm
 * Signature: (IIIIIIF[FII[FIIF[FII)I
 */
JNIEXPORT jint JNICALL Java_org_bitbucket_ioplus_clblas_BLAS_sgemm(
                                             JNIEnv * env, jclass class,
                                             jint order,
                                             jint  transA,
                                             jint transB,
                                             jint m,
                                             jint n,
                                             jint k,
                                             jfloat alpha,
                                             jfloatArray a,
                                             jint offA,
                                             jint lda,
                                             jfloatArray b,
                                             jint offB,
                                             jint ldb,
                                             jfloat beta,
                                             jfloatArray c,
                                             jint offC,
                                             jint ldc
                                             )
{

    cl_int err;
    cl_mem bufA, bufB, bufC;
    cl_event event = NULL;


    err = 0;

    (*env)->MonitorEnter(env, a);
    (*env)->MonitorEnter(env, b);
    (*env)->MonitorEnter(env, c);


    jfloat *aPtrBase = 0, *aPtr = 0;
    if (a) {
        aPtrBase = (*env)->GetFloatArrayElements(env, a, NULL);
        aPtr = aPtrBase + offA;
    }
    jfloat *bPtrBase = 0, *bPtr = 0;
    if (b) {
        if((*env)->IsSameObject(env, b, a) == JNI_TRUE)
            bPtrBase = aPtrBase;
        else
            bPtrBase = (*env)->GetFloatArrayElements(env, b, NULL);
        bPtr = bPtrBase + offB;
    }
    jfloat *cPtrBase = 0, *cPtr = 0;
    if (c) {
        if((*env)->IsSameObject(env, c, a) == JNI_TRUE)
            cPtrBase = aPtrBase;
        else
            if((*env)->IsSameObject(env, c, b) == JNI_TRUE)
                cPtrBase = bPtrBase;
            else
                cPtrBase = (*env)->GetFloatArrayElements(env, c, NULL);
        cPtr = cPtrBase + offC;
    }

    jint A_len = (*env)->GetArrayLength(env,a) - offA;
    jint B_len = (*env)->GetArrayLength(env,b) - offB;
    jint C_len = (*env)->GetArrayLength(env,c) - offC;





    /* Prepare OpenCL memory objects and place matrices inside them. */
    bufA = clCreateBuffer(ctx, CL_MEM_READ_ONLY, A_len * sizeof(*aPtr),
                          NULL, &err);
    bufB = clCreateBuffer(ctx, CL_MEM_READ_ONLY, B_len * sizeof(*bPtr),
                          NULL, &err);
    bufC = clCreateBuffer(ctx, CL_MEM_READ_WRITE, C_len * sizeof(*cPtr),
                          NULL, &err);

    err = clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0,
                              A_len * sizeof(*aPtr), aPtr, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0,
                               B_len * sizeof(*bPtr), bPtr, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, bufC, CL_TRUE, 0,
                               C_len * sizeof(*cPtr), cPtr, 0, NULL, NULL);

    /* Call clblas extended function. */
    err = clblasSgemm((clblasOrder)order,
                      transA, transB,
                      m,n,k,
                      alpha, bufA, 0, lda,
                      bufB, 0, ldb, beta,
                      bufC, 0, ldc,
                      1, &queue, 0, NULL, &event);
    if (err != CL_SUCCESS) {
        printf("clblasSgemmEx() failed with %d\n", err);
        return 1;
    }
    else {
        /* Wait for calculations to be finished. */
        err = clWaitForEvents(1, &event);

        /* Fetch results of calculations from GPU memory. */
        err = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0,
                                  C_len * sizeof(*cPtr),
                                  cPtr, 0, NULL, NULL);
    }

    /* Release OpenCL memory objects. */
    clReleaseMemObject(bufC);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufA);


    if(cPtrBase) {
        (*env)->ReleaseFloatArrayElements(env, c, cPtrBase, 0);
        if (cPtrBase == aPtrBase)
            aPtrBase = 0;
        if (cPtrBase == bPtrBase)
            bPtrBase = 0;
        cPtrBase = 0;
    }
    if(bPtrBase) {
        (*env)->ReleaseFloatArrayElements(env, b, bPtrBase, JNI_ABORT);
        if (bPtrBase == aPtrBase)
            aPtrBase = 0;
        bPtrBase = 0;
    }
    if(aPtrBase) {
        (*env)->ReleaseFloatArrayElements(env, a, aPtrBase, JNI_ABORT);
        aPtrBase = 0;
    }

    (*env)->MonitorExit(env, a);
    (*env)->MonitorExit(env, b);
    (*env)->MonitorExit(env, c);

    //printf("clblasSgemm() done.\n");


    return err;
}

/* /\* */
/*  * Class:     org_bitbucket_ioplus_clblas_BLAS */
/*  * Method:    sgemm */
/*  * Signature: (IIIIIIF[FII[FIIF[FII)I */
/*  *\/ */
/* JNIEXPORT jint JNICALL Java_org_bitbucket_ioplus_clblas_BLAS_sgemm( */
/*                                              JNIEnv * env, jclass class, */
/*                                              jint order, */
/*                                              jint  transA, */
/*                                              jint transB, */
/*                                              jint m, */
/*                                              jint n, */
/*                                              jint k, */
/*                                              jfloat alpha, */
/*                                              jfloatArray a, */
/*                                              jint offA, */
/*                                              jint lda, */
/*                                              jfloatArray b, */
/*                                              jint offB, */
/*                                              jint ldb, */
/*                                              jfloat beta, */
/*                                              jfloatArray c, */
/*                                              jint offC, */
/*                                              jint ldc */
/*                                              ) */
/* { */

/*     cl_int err; */
/*     cl_mem bufA, bufB, bufC; */
/*     cl_event event = NULL; */


/*     int M = m, N = n, K = k; */

/*     err = 0; */

/*     /\* jni arguments *\/ */

/*     (*env)->MonitorEnter(env, a); */
/*     (*env)->MonitorEnter(env, b); */
/*     (*env)->MonitorEnter(env, c); */

/*     cl_float * A = 0; */
/*     cl_float * B = 0; */
/*     cl_float * C = 0; */
/*     A = (*env)->GetPrimitiveArrayCritical(env, a, 0); */
/*     B = (*env)->GetPrimitiveArrayCritical(env, b, 0); */
/*     C = (*env)->GetPrimitiveArrayCritical(env, c, 0); */



/* //    A = (*env)->GetFloatArrayElements(env, a, NULL); */
/* //    B = (*env)->GetFloatArrayElements(env, b, NULL); */
/* //    C = (*env)->GetFloatArrayElements(env, c, NULL); */



/*     jint A_len = (*env)->GetArrayLength(env,a); */
/*     jint B_len = (*env)->GetArrayLength(env,b); */
/*     jint C_len = (*env)->GetArrayLength(env,c); */





/*     /\* Prepare OpenCL memory objects and place matrices inside them. *\/ */
/*     bufA = clCreateBuffer(ctx, CL_MEM_READ_ONLY, A_len * sizeof(*A), */
/*                           NULL, &err); */
/*     bufB = clCreateBuffer(ctx, CL_MEM_READ_ONLY, B_len * sizeof(*B), */
/*                           NULL, &err); */
/*     bufC = clCreateBuffer(ctx, CL_MEM_READ_WRITE, C_len * sizeof(*C), */
/*                           NULL, &err); */

/*     err = clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, */
/*                               A_len * sizeof(*A), A, 0, NULL, NULL); */
/*     err = clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, */
/*                                B_len * sizeof(*B), B, 0, NULL, NULL); */
/*     err = clEnqueueWriteBuffer(queue, bufC, CL_TRUE, 0, */
/*                                C_len * sizeof(*C), C, 0, NULL, NULL); */

/*     /\* Call clblas extended function. *\/ */
/*     err = clblasSgemm((clblasOrder)order, */
/*                       transA, transB, */
/*                       M, N, K, */
/*                       alpha, bufA, offA, lda, */
/*                       bufB, offB, ldb, beta, */
/*                       bufC, offC, ldc, */
/*                       1, &queue, 0, NULL, &event); */
/*     if (err != CL_SUCCESS) { */
/*         printf("clblasSgemmEx() failed with %d\n", err); */
/*         return 1; */
/*     } */
/*     else { */
/*         /\* Wait for calculations to be finished. *\/ */
/*         err = clWaitForEvents(1, &event); */

/*         /\* Fetch results of calculations from GPU memory. *\/ */
/*         err = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, */
/*                                   C_len * sizeof(*C), */
/*                                   C, 0, NULL, NULL); */
/*     } */

/*     /\* Release OpenCL memory objects. *\/ */
/*     clReleaseMemObject(bufC); */
/*     clReleaseMemObject(bufB); */
/*     clReleaseMemObject(bufA); */


/*     (*env)->ReleasePrimitiveArrayCritical(env, a, A, 0); */
/*     (*env)->ReleasePrimitiveArrayCritical(env, b, B, 0); */
/*     (*env)->ReleasePrimitiveArrayCritical(env, c, C, 0); */

/* // */
/* //    (*env)->ReleaseFloatArrayElements(env, c, C, 0); */
/* //    (*env)->ReleaseFloatArrayElements(env, a, A, JNI_ABORT); */
/* //    (*env)->ReleaseFloatArrayElements(env, b, B, JNI_ABORT); */

/*     (*env)->MonitorExit(env, a); */
/*     (*env)->MonitorExit(env, b); */
/*     (*env)->MonitorExit(env, c); */

/*     C = 0; */
/*     A = 0; */
/*     B = 0; */
/*     //printf("clblasSgemm() done.\n"); */


/*     return err; */
/* } */

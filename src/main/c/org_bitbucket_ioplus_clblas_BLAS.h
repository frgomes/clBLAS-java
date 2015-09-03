/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class org_bitbucket_ioplus_clblas_BLAS */

#ifndef _Included_org_bitbucket_ioplus_clblas_BLAS
#define _Included_org_bitbucket_ioplus_clblas_BLAS
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     org_bitbucket_ioplus_clblas_BLAS
 * Method:    setup
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_org_bitbucket_ioplus_clblas_BLAS_setup
  (JNIEnv *, jclass);

/*
 * Class:     org_bitbucket_ioplus_clblas_BLAS
 * Method:    tearDown
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_org_bitbucket_ioplus_clblas_BLAS_tearDown
  (JNIEnv *, jclass);



/*
 * Class:     org_bitbucket_ioplus_clblas_BLAS
 * Method:    sscal
 * Signature: (IF[FII)V
 */
JNIEXPORT void JNICALL Java_org_bitbucket_ioplus_clblas_BLAS_sscal
  (JNIEnv *, jclass, jint, jfloat, jfloatArray, jint, jint);

/*
 * Class:     org_bitbucket_ioplus_clblas_BLAS
 * Method:    saxpy
 * Signature: (IF[FII[FII)V
 */
JNIEXPORT void JNICALL Java_org_bitbucket_ioplus_clblas_BLAS_saxpy
  (JNIEnv *, jclass, jint, jfloat, jfloatArray, jint, jint, jfloatArray, jint, jint);

/*
 * Class:     org_bitbucket_ioplus_clblas_BLAS
 * Method:    sdot
 * Signature: (I[FI[FII[FII)V
 */
JNIEXPORT void JNICALL Java_org_bitbucket_ioplus_clblas_BLAS_sdot
  (JNIEnv *, jclass, jint, jfloatArray, jint, jfloatArray, jint, jint, jfloatArray, jint, jint);

/*
 * Class:     org_bitbucket_ioplus_clblas_BLAS
 * Method:    scopy
 * Signature: (I[FII[FII)V
 */
JNIEXPORT void JNICALL Java_org_bitbucket_ioplus_clblas_BLAS_scopy
  (JNIEnv *, jclass, jint, jfloatArray, jint, jint, jfloatArray, jint, jint);



/*
 * Class:     org_bitbucket_ioplus_clblas_BLAS
 * Method:    sgemv
 * Signature: (IIIIF[FII[FIIF[FII)I
 */
JNIEXPORT jint JNICALL Java_org_bitbucket_ioplus_clblas_BLAS_sgemv
  (JNIEnv *, jclass, jint, jint, jint, jint, jfloat, jfloatArray, jint, jint, jfloatArray, jint, jint, jfloat, jfloatArray, jint, jint);

/*
 * Class:     org_bitbucket_ioplus_clblas_BLAS
 * Method:    sger
 * Signature: (IIIF[FII[FII[FII)I
 */
JNIEXPORT jint JNICALL Java_org_bitbucket_ioplus_clblas_BLAS_sger
  (JNIEnv *, jclass, jint, jint, jint, jfloat, jfloatArray, jint, jint, jfloatArray, jint, jint, jfloatArray, jint, jint);


/*
 * Class:     org_bitbucket_ioplus_clblas_BLAS
 * Method:    sgemm
 * Signature: (IIIIIIF[FII[FIIF[FII)I
 */
    JNIEXPORT jint JNICALL Java_org_bitbucket_ioplus_clblas_BLAS_sgemm
    (JNIEnv *, jclass, jint, jint, jint, jint, jint, jint, jfloat, jfloatArray, jint, jint, jfloatArray, jint, jint, jfloat, jfloatArray, jint, jint);

#ifdef __cplusplus
}
#endif
#endif
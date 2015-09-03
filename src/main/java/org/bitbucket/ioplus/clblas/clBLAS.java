package org.bitbucket.ioplus.clblas;

import org.bitbucket.ioplus.nativeloader.NativeLoader;

import java.io.IOException;

/**
 * Created by przemek on 11/08/15.
 */
public class clBLAS {

    private clBLAS(){}

    static {
        try {
            NativeLoader.load(clBLAS.class);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}

#!/bin/sh
gcc --shared -fPIC BLAS.c -L/opt/clBLAS-2.6.0-Linux-x64/lib64 -lm -lclBLAS -lOpenCL -lpthread  -o ../resources/native/Linux/amd64/libBLAS.so

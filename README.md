# clBLAS-java
Java native interface (JNI) wrapper for OpenCL BLAS implementation optimized for AMD hardware (GPU, APU &amp; CPU fallback).




### Requirements

This library bridges clMathLibraries/clBLAS and java through jni calls.
Original C code is published in [clMathLibraries repository](https://github.com/clMathLibraries/clBLAS).

clBLAS-java ships with a shared library for x86_64 Linux, which is automatically loaded to jvm at a runtime, but you should have installed ICD loader in your system to make it work.
Code has been tested on Linux Mint 17 with Devastator and AMD A8-4500M APU with Radeon(tm) HD Graphics and using AMD proprietary driver but, in theory, it should also work on any hardware with OpenCl driver.


### Setup

This is Java 8 project. No other java dependencies are needed.

While immature, project releases and snapshots are available in a maven repository which is located on corresponding branches.



To use it in your maven project, setup repositories directly in pom.xml:


```xml
<repositories>
        <repository>
                <id>clBLAS-java</id>
                <name>clBLAS-java releases</name>
                <url>https://raw.github.com/sidec/clBLAS-java/releases</url>
        </repository>
        <repository>
                <id>clBLAS-java</id>
                <name>clBLAS-java snapshots</name>
                <url>https://raw.github.com/sidec/clBLAS-java/snapshots</url>
        </repository>
</repositories>
```

Then you can add a dependency to your project:

```xml
<dependency>
          <groupId>org.bitbucket.ioplus</groupId>
          <artifactId>clBLAS-java</artifactId>
          <version>0.0.2-SNAPSHOT</version>
</dependency>
```

Or, for the latest release:

```xml
<dependency>
          <groupId>org.bitbucket.ioplus</groupId>
          <artifactId>clBLAS-java</artifactId>
          <version>0.0.1</version>
</dependency>
```

### Documentation

clBLAS is documented [here](http://clmathlibraries.github.io/clBLAS).

On Java side, only a small subset of BLAS Level 1,2,3 is implemented. However, adding more functionality should be straightforward.

Code base is relatively small, so please read the source at the moment. Also, tests might be helpfull.

## Pull requests are highly welcome!

Just fork or clone it:

```sh
git clone https://github.com/sidec/clBLAS-java.git
```

Add new feature or fix bugs, test it, and install locally with 

```sh
mvn clean compile install
```


package org.bitbucket.ioplus.nativeloader;

import java.io.*;
import java.net.URL;
import java.util.UUID;

/**
 * Created by przemek on 09/08/15.
 */
public class NativeLoader {

    private NativeLoader(){}

    public static void load(Class c) throws IOException {

        String libraryName = System.mapLibraryName(c.getSimpleName());

        String osName = System.getProperty("os.name");
        String osArch = System.getProperty("os.arch");
        String path = File.separator + "native" +
                File.separator + osName +
                File.separator + osArch +
                File.separator + libraryName;
        System.out.println(path);

        String tmpDir = System.getProperty("java.io.tmpdir");
        final String id = tmpDir + File.separator + UUID.randomUUID().toString() + libraryName;
        File temp = new File(id);
        temp.createNewFile();
        temp.deleteOnExit();

        System.out.println("Loading "+ libraryName +" from temp copy: " + temp.getAbsolutePath());



        byte[] buffer = new byte[1024];
        int readBytes;

        try (InputStream is = NativeLoader.class.getResourceAsStream(path);
             OutputStream os = new FileOutputStream(temp)){
            if(is != null) {
                while ((readBytes = is.read(buffer)) != -1) {
                    os.write(buffer, 0, readBytes);
                }
            } else {
                throw new FileNotFoundException("File " + path + " was not found inside JAR.");
            }
        }


        System.load(temp.getAbsolutePath());
        System.out.println("Library loaded.");

        Runtime.getRuntime().addShutdownHook(new Thread() {
            @Override
            public void run() {
                File toDelete = new File(id);
                if(toDelete.exists()){
                    toDelete.delete();
                    System.out.println("File " + id + " was deleted.");
                }

            }
        });

    }



}

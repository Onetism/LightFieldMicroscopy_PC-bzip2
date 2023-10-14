# Light field File Format (.lfm) - JNI Library #

## Java Native Interface (JNI) ##

The LFM API is exposed on the Java-side through a JNI wrapper, included in the "javaWrapper" subfolder. It can be build with maven, includes compiled native libraries for Windows and Linux (both 64-bit) and will eventually be available as an artifact on a Maven repository. ImageJ users on supported platforms can simply install LFM support by following the update site (see below).

## Build JNI library from source
  - install Maven
  - navigate to the javaWrapper subfolder
  - run "mvn clean package"
  - the JAR file will be built at "javaWrapper/target/lfm-[version].jar"

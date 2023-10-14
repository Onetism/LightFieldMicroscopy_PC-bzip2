find_package(JNI REQUIRED)

# Install the JNI product to the correct resource path
# The paths are relative to the  CMAKE_PREFIX_PATH which should be set at
# config time.
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(nativedir x86_64-max_os_x)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(nativedir amd64-linux)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(nativedir amd64-windows)
else()
    set(nativedir unknown)
    message(WARNING "Could not detect platform for installing the JNI interface.")
endif()

set(tgt lfm-jni)
add_library(${tgt} SHARED org_janelia_simview_lfm_LFMJNI.cpp)
target_include_directories(${tgt} PUBLIC ${JNI_INCLUDE_DIRS})
target_link_libraries (${tgt} lfm_static)
install(TARGETS ${tgt} DESTINATION
        ${PROJECT_SOURCE_DIR}/javaWrapper/src/main/resources/native/${nativedir})

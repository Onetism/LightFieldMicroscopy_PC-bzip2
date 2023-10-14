add_subdirectory(bzip2-1.0.6)
# add_subdirectory(cuda_bzip2)

set(SKIP_INSTALL_ALL       TRUE)
set(SKIP_INSTALL_EXPORT    TRUE)
set(SKIP_TESTS             TRUE)
add_subdirectory(zlib-1.2.8)

set_target_properties(zlib PROPERTIES EXCLUDE_FROM_ALL TRUE EXCLUDE_FROM_DEFAULT_BUILD TRUE)

# organize things for the IDE
set_target_properties(
        bzip2
        bzip2s
        # cuda_bzip2
        zlib
        zlibstatic
    PROPERTIES
        FOLDER external
)

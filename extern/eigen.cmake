# Export target:
# - ExternEigen
#
# Export variables:
# - EIGEN_INCLUDE_DIRS

include(ExternalProject)

ExternalProject_Add(
    extern_eigen
    PREFIX eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/da7909592376c893dabbc4b6453a8ffe46b1eb8e/eigen-da7909592376c893dabbc4b6453a8ffe46b1eb8e.tar.gz
    URL_HASH SHA256=37f71e1d7c408e2cc29ef90dcda265e2de0ad5ed6249d7552f6c33897eafd674
    DOWNLOAD_DIR "${EXTERN_DOWNLOAD_DIR}/eigen"
    UPDATE_COMMAND ""
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DCMAKE_BUILD_TYPE=Release
)

ExternalProject_Get_Property(extern_eigen INSTALL_DIR)
set(EIGEN_INCLUDE_DIRS "${INSTALL_DIR}/include/eigen3")

add_library(ExternEigen INTERFACE)
add_dependencies(ExternEigen extern_eigen)
target_compile_definitions(ExternEigen INTERFACE
    -DDISABLE_PCAP
    -DDISABLE_PNG
)
target_include_directories(ExternEigen INTERFACE ${EIGEN_INCLUDE_DIRS})

include(FetchContent)

FetchContent_Declare(
    extern_pybind11
    PREFIX pybind11
    URL https://github.com/pybind/pybind11/archive/refs/tags/v2.10.1.tar.gz
    URL_HASH SHA256=111014b516b625083bef701df7880f78c2243835abdb263065b6b59b960b6bad
    DOWNLOAD_DIR "${EXTERN_DOWNLOAD_DIR}/pybind11"
)

FetchContent_MakeAvailable(extern_pybind11)

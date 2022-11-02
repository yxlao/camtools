cmake_minimum_required(VERSION 3.16)

file(STRINGS "camtools/version.py" READ_VERSION)
foreach(line ${READ_VERSION})
    if (line MATCHES "__version__ = \"(.*)\"")
        set(camtools_version ${CMAKE_MATCH_1})
        break()
    endif()
endforeach()
if (camtools_version)
    message(STATUS "Parsed version: ${camtools_version}")
else()
    message(FATAL_ERROR "Could not parse version from camtools/version.py")
endif()

project(camtools
    VERSION ${camtools_version}
    DESCRIPTION "Camtools: Camera Tools for Computer Vision."
    LANGUAGES CXX
)

set(CMAKE_BUILD_TYPE "RelWithDebInfo")

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib/$<CONFIG>)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib/$<CONFIG>)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

set(EXTERN_DOWNLOAD_DIR "${PROJECT_SOURCE_DIR}/extern_downloads")

add_subdirectory(extern)
add_subdirectory(cpp)

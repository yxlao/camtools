find_package(Python3 3.7 COMPONENTS Interpreter Development) # 3.7+
if (Python3_FOUND)
    message(STATUS "Using Python version: ${Python3_EXECUTABLE}")
else()
    message(FATAL_ERROR "Python 3.7+ not found.")
endif()


pybind11_add_module(camtools_cpp)
target_sources(camtools_cpp PRIVATE
    pointcloud_cropper.cpp
)
target_link_directories(camtools_cpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(camtools_cpp PRIVATE
    ExternTBB
)

set(python_package_dir "${PROJECT_BINARY_DIR}/python_package")

add_custom_target(wheel ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory                                        "${python_package_dir}"
    COMMAND ${CMAKE_COMMAND} -E rm -rf                                                "${python_package_dir}/camtools"
    COMMAND ${CMAKE_COMMAND} -E copy_directory    "${PROJECT_SOURCE_DIR}/camtools"    "${python_package_dir}/camtools"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:camtools_cpp>"    "${python_package_dir}/camtools"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${PROJECT_SOURCE_DIR}/setup.py"    "${python_package_dir}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${PROJECT_SOURCE_DIR}/MANIFEST.in" "${python_package_dir}"
    COMMAND ${CMAKE_COMMAND} -E chdir "${python_package_dir}" ${Python3_EXECUTABLE} setup.py bdist_wheel
    DEPENDS camtools_cpp
)

add_custom_target(pip-install
    COMMAND ${Python3_EXECUTABLE} -m pip install --upgrade --force-reinstall --no-index --find-links "." camtools
    DEPENDS wheel
    WORKING_DIRECTORY "${python_package_dir}/dist"
)

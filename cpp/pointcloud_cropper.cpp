#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "stream_compact.h"

namespace camtools {

namespace py = pybind11;
using namespace py::literals;

template <class T>
struct Point {
    T vals[3];
};

static py::array ToIntArray1D(const int *array_buf, int N) {
    py::dtype py_dtype(py::format_descriptor<int32_t>::format());
    py::array::ShapeContainer py_shape({N});
    py::array::StridesContainer py_strides({4});  // int32_t is 4 bytes

    auto py_destructor = [](PyObject *data) {
        // Deallocates buffer when the numpy variable goes out of scope.
        int *buf = reinterpret_cast<int *>(
                PyCapsule_GetPointer(data, "int32_t buffer"));
        if (buf) {
            // std::cout << "destructed" << std::endl;
            delete[] buf;
        } else {
            PyErr_Clear();
        }
    };
    py::capsule py_capsule(array_buf, "int32_t buffer", py_destructor);
    return py::array(py_dtype, py_shape, py_strides, array_buf, py_capsule);
}

template <typename T>
static py::array GetCropIndex(py::array_t<T> points,
                              double x_min,
                              double y_min,
                              double z_min,
                              double x_max,
                              double y_max,
                              double z_max) {
    // Parse input buffer as array of Point<T>.
    py::buffer_info info = points.request();
    std::vector<int> shape(info.shape.begin(), info.shape.end());
    if (shape.size() != 2 || shape[1] != 3) {
        throw std::runtime_error("points must be (N, 3)");
    }
    int N = shape[0];
    const Point<T> *points_ptr = reinterpret_cast<const Point<T> *>(info.ptr);

    // Output buffer will be freed by py::capsule.
    int *valid_indices = new int[N];

    // Perform stream compaction.
    auto is_valid = [&](const Point<T> &point) -> bool {
        return point.vals[0] >= x_min && point.vals[0] <= x_max &&
               point.vals[1] >= y_min && point.vals[1] <= y_max &&
               point.vals[2] >= z_min && point.vals[2] <= z_max;
    };
    int num_valid = GetValidIndices(points_ptr, valid_indices, N, is_valid);
    return ToIntArray1D(valid_indices, num_valid);
}

template <typename T>
static py::array GetCropIndexSerial(py::array_t<T> points,
                                    double x_min,
                                    double y_min,
                                    double z_min,
                                    double x_max,
                                    double y_max,
                                    double z_max) {
    // Parse input buffer as array of Point<T>.
    py::buffer_info info = points.request();
    std::vector<int> shape(info.shape.begin(), info.shape.end());
    if (shape.size() != 2 || shape[1] != 3) {
        throw std::runtime_error("points must be (N, 3)");
    }
    int N = shape[0];
    const Point<T> *points_ptr = reinterpret_cast<const Point<T> *>(info.ptr);

    // Output buffer will be freed by py::capsule.
    int *valid_indices = new int[N];

    // Perform stream compaction.
    auto is_valid = [&](const Point<T> &point) -> bool {
        return point.vals[0] >= x_min && point.vals[0] <= x_max &&
               point.vals[1] >= y_min && point.vals[1] <= y_max &&
               point.vals[2] >= z_min && point.vals[2] <= z_max;
    };
    int num_valid =
            GetValidIndicesSerial(points_ptr, valid_indices, N, is_valid);
    return ToIntArray1D(valid_indices, num_valid);
}

PYBIND11_MODULE(camtools_cpp, m) {
    m.def("get_crop_indices", &GetCropIndex<double>, "points"_a, "x_min"_a,
          "y_min"_a, "z_min"_a, "x_max"_a, "y_max"_a, "z_max"_a);
    m.def("get_crop_indices", &GetCropIndex<float>, "points"_a, "x_min"_a,
          "y_min"_a, "z_min"_a, "x_max"_a, "y_max"_a, "z_max"_a);
    m.def("get_crop_indices_serial", &GetCropIndexSerial<double>, "points"_a,
          "x_min"_a, "y_min"_a, "z_min"_a, "x_max"_a, "y_max"_a, "z_max"_a);
    m.def("get_crop_indices_serial", &GetCropIndexSerial<float>, "points"_a,
          "x_min"_a, "y_min"_a, "z_min"_a, "x_max"_a, "y_max"_a, "z_max"_a);
}

}  // namespace camtools

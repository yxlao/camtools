#pragma once

#include <tbb/parallel_for.h>
#include <tbb/parallel_scan.h>

#include <functional>

namespace camtools {

// General purpose parallelized stream compaction with TBB.
template <typename dtype_t, typename index_t, typename func_t>
index_t GetValidIndices(const dtype_t* src,
                        index_t* valid_indices,
                        index_t N,
                        func_t is_valid) {
    index_t num_valid = tbb::parallel_scan(
            tbb::blocked_range<index_t>(0, N), 0,
            [&](const tbb::blocked_range<index_t>& r, index_t sum,
                bool is_final_scan) -> index_t {
                index_t temp = sum;
                for (index_t i = r.begin(); i < r.end(); ++i) {
                    if (is_valid(src[i])) {
                        if (is_final_scan) {
                            valid_indices[temp] = i;
                        }
                        temp += 1;
                    }
                }
                return temp;
            },
            [](int left, int right) { return left + right; });
    return num_valid;
}

// General purpose stream compaction serial implementation.
template <typename dtype_t, typename index_t, typename func_t>
index_t GetValidIndicesSerial(const dtype_t* src,
                              index_t* valid_indices,
                              index_t N,
                              func_t is_valid) {
    index_t num_valid = 0;
    for (index_t i = 0; i < N; i++) {
        if (is_valid(src[i])) {
            valid_indices[num_valid++] = i;
        }
    }
    return num_valid;
}

}  // namespace camtools

#include <cuda_runtime.h>  // Include for cudaStream_t

inline void export_kernel(py::module &m) { 
    m.def("gspmmv", [](graph_t& graph, py::capsule& input1, py::capsule& output, bool reverse, bool norm, uint64_t stream_handle) {
        // Convert PyTorch tensors (via capsule) to array2d_t
        array2d_t<float> input1_array = capsule_to_array2d(input1);
        array2d_t<float> output_array = capsule_to_array2d(output);

        // Cast the stream handle to cudaStream_t
        cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);

        // Call the actual gspmmv function
        return gspmmv(graph, input1_array, output_array, reverse, norm, stream);
    });
}

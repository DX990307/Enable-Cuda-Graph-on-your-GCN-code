#pragma once
#include "csr.h"
#include "op.h"
#include <cuda_runtime.h>  // Include for cudaStream_t

void gspmmv(graph_t& graph, array2d_t<float>& input1, array2d_t<float>& output, bool reverse, bool norm, cudaStream_t stream);
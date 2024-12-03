import torch as th
import torch.utils.dlpack
import graphpy as gpk

def gp_gspmmv(graph, input1, dim1_0, dim1_1, reverse, norm, device0):
    # Get the current PyTorch CUDA stream
    stream = th.cuda.current_stream(device=device0)
    stream_handle = stream.cuda_stream

    input1_dl = th.utils.dlpack.to_dlpack(input1)
    res1 = th.zeros(dim1_0, dim1_1, device=device0)
    res_dl1 = th.utils.dlpack.to_dlpack(res1)

    # Pass the CUDA stream to the backend
    gpk.gspmmv(graph, input1_dl, res_dl1, reverse, norm, stream_handle)
    return res1

import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal


"""
A fused convolution - maxpool kernel that you need to implement for Part 2.

Parameters:
    X: the input tensor
    W: the weights of the convolution filters.
    bias: the biases of the convolution filters.
    pool_size: the size of the pool filter and pool stride.

expect: X.shape == [batch_size, in_channels, input_height, input_width]
expect: W.shape == [out_channels, in_channels, filter_height, filter_width]
expect: bias.shape == [out_channels]
expect: filter_height == filter_width
expect: pool_size == 1 || pool_size == 2
expect: input_channels % 128 == 0
expect: output_channels % 128 == 0

out_height = input_height - filter_height + 1
out_width = input_width - filter_width + 1

out_pool_height = out_height // pool_size
out_pool_width = out_width // pool_size

The shape of the output should be [batch_size, out_channels, out_pool_height, out_pool_width]

"""
@nki.jit
def fused_conv2d_maxpool(X, W, bias, pool_size=1):
    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]

    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    out_pool_height = out_height // pool_size
    out_pool_width = out_width // pool_size
   
    assert in_channels % 128 == 0

    assert nl.tile_size.gemm_moving_fmax >= out_width

    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    c_in_pmax = nl.tile_size.pmax
    n_tiles_c_in = in_channels // c_in_pmax
    c_out_pmax = c_in_pmax
    n_tiles_c_out = out_channels // c_out_pmax

    out_chunks = 2
    in_rows = out_chunks + filter_height - 1
    num_output_chunks = (out_height + out_chunks - 1) // out_chunks

    W = W.reshape((n_tiles_c_out, c_out_pmax, n_tiles_c_in, c_in_pmax, filter_height, filter_width))
    weight_sbuf = nl.ndarray((n_tiles_c_out, nl.par_dim(c_out_pmax), n_tiles_c_in, c_in_pmax, filter_height, filter_width), dtype=W.dtype, buffer=nl.sbuf)
    weight_copy = nl.ndarray((filter_height, filter_width, n_tiles_c_out, n_tiles_c_in, nl.par_dim(c_out_pmax), c_in_pmax), dtype=W.dtype, buffer=nl.sbuf)
    w = nl.ndarray((filter_height, filter_width, n_tiles_c_out, n_tiles_c_in, nl.par_dim(c_in_pmax), c_out_pmax), dtype=W.dtype, buffer=nl.sbuf)

    for c_out_tile in nl.affine_range(n_tiles_c_out):
        weight_sbuf[c_out_tile] = nl.load(W[c_out_tile])
   
    for c_out_tile in nl.affine_range(n_tiles_c_out):
        for c_in_tile in nl.affine_range(n_tiles_c_in):
            for i in nl.affine_range(filter_height):
                for j in nl.affine_range(filter_width):
                    weight_copy[i, j, c_out_tile, c_in_tile, :, :] = nl.copy(weight_sbuf[c_out_tile, :, c_in_tile, :, i, j], dtype=W.dtype)
                    w[i, j, c_out_tile, c_in_tile] = nisa.nc_transpose(weight_copy[i, j, c_out_tile, c_in_tile])
    
    # Process the image in batches
    for b in nl.affine_range(batch_size):
        for n in nl.affine_range(num_output_chunks):
            x = nl.ndarray(
                shape=(n_tiles_c_in, nl.par_dim(c_in_pmax), in_rows, input_width), 
                dtype=X.dtype,
                buffer=nl.sbuf
            )
            for c_in_tile in nl.affine_range(n_tiles_c_in):
                x[c_in_tile] = nl.load(X[b, c_in_tile*c_in_pmax:(c_in_tile+1)*c_in_pmax, n*(out_chunks):(n + 1)*out_chunks + 2]) # adjust starting point (in_rows*n)
            for c_out_tile in nl.affine_range(n_tiles_c_out):
                output_sbuf = nl.ndarray(
                    shape=(nl.par_dim(c_out_pmax), out_chunks, out_width), 
                    dtype=X.dtype,
                    buffer=nl.sbuf
                )
                bias_tile = nl.ndarray(shape = (nl.par_dim(c_out_pmax), ), dtype = bias.dtype, buffer=nl.sbuf)
                bias_tile = nl.load(bias[c_out_tile*c_out_pmax:(c_out_tile + 1)*c_out_pmax])
                for out_row in nl.affine_range(out_chunks): 
                    result = nl.zeros((128, out_width), nl.float32, buffer=nl.psum)
                    for i in nl.affine_range(filter_height):
                        for j in nl.affine_range(filter_width):
                            for c_in_tile in nl.affine_range(n_tiles_c_in):
                                result += nl.matmul(
                                    w[i, j, c_out_tile, c_in_tile, :, : ],
                                    x[c_in_tile, :, out_row + i, j:j + out_width],
                                    transpose_x=True
                                )

                    result = nisa.tensor_scalar(result, np.add, bias_tile)
                    output_sbuf[:, out_row, :] = nl.copy(result, dtype=X.dtype)

                nl.store(X_out[b, c_out_tile*c_out_pmax:(c_out_tile+1)*c_out_pmax, n*out_chunks:(n + 1)*out_chunks], output_sbuf)
    return X_out

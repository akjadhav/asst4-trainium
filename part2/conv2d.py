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
    out_channels, in_channels_check, filter_height, filter_width = W.shape
    bias_dim = bias.shape[0]

    assert in_channels_check == in_channels and bias_dim == out_channels, (
        f"Input/weight/bias shape mismatch: "
        f"in_channels={in_channels}, in_channels_check={in_channels_check}, "
        f"out_channels={out_channels}, bias_dim={bias_dim}"
    )

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    out_pool_height = out_height // pool_size
    out_pool_width = out_width // pool_size
    
    assert in_channels % 128 == 0
    assert nl.tile_size.gemm_moving_fmax >= out_width

    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm
    )

    c_in_block = nl.tile_size.pmax
    n_tiles_c_in = in_channels // c_in_block
    c_out_block = c_in_block
    n_tiles_c_out = out_channels // c_out_block

    W = W.reshape((n_tiles_c_out, c_out_block, n_tiles_c_in, c_in_block, filter_height, filter_width))
    w_sbuf = nl.ndarray(
        (n_tiles_c_out, nl.par_dim(c_out_block), n_tiles_c_in, c_in_block, filter_height, filter_width),
        dtype=W.dtype, buffer=nl.sbuf
    )
    w_temp = nl.ndarray(
        (filter_height, filter_width, n_tiles_c_out, n_tiles_c_in, nl.par_dim(c_out_block), c_in_block), 
        dtype=W.dtype, buffer=nl.sbuf
    )
    w_trans = nl.ndarray(
        (filter_height, filter_width, n_tiles_c_out, n_tiles_c_in, nl.par_dim(c_in_block), c_out_block),
        dtype=W.dtype, buffer=nl.sbuf
    )

    for oc_tile in nl.affine_range(n_tiles_c_out):
        w_sbuf[oc_tile] = nl.load(W[oc_tile])

    for oc_tile in nl.affine_range(n_tiles_c_out):
        for ic_tile in nl.affine_range(n_tiles_c_in):
            for fh in nl.affine_range(filter_height):
                for fw in nl.affine_range(filter_width):
                    w_temp[fh, fw, oc_tile, ic_tile, :, :] = nl.copy(
                        w_sbuf[oc_tile, :, ic_tile, :, fh, fw],
                        dtype=W.dtype
                    )
                    w_trans[fh, fw, oc_tile, ic_tile] = nisa.nc_transpose(
                        w_temp[fh, fw, oc_tile, ic_tile]
                    )

    chunk_size = 2
    total_chunks = (out_height + chunk_size - 1) // chunk_size

    input_rows_chunk = chunk_size + filter_height - 1

    for b_idx in nl.affine_range(batch_size):
        for chunk_idx in nl.affine_range(total_chunks):

            x_sbuf = nl.ndarray(
                shape=(n_tiles_c_in, nl.par_dim(c_in_block), input_rows_chunk, input_width),
                dtype=X.dtype,
                buffer=nl.sbuf
            )

            start_in_row = chunk_idx * chunk_size
            end_in_row = start_in_row + input_rows_chunk
            for ic_tile in nl.affine_range(n_tiles_c_in):
                x_sbuf[ic_tile] = nl.load(X[b_idx, ic_tile*c_in_block:(ic_tile+1)*c_in_block, start_in_row:end_in_row])

            for oc_tile in nl.affine_range(n_tiles_c_out):
                y_sbuf = nl.ndarray(
                    (nl.par_dim(c_out_block), chunk_size, out_width),
                    dtype=X.dtype, buffer=nl.sbuf
                )
                bias_sbuf = nl.ndarray(
                    (nl.par_dim(c_out_block),), dtype=bias.dtype, buffer=nl.sbuf
                )
                bias_sbuf = nl.load(bias[oc_tile*c_out_block:(oc_tile+1)*c_out_block])

                for out_r in nl.affine_range(chunk_size):
                    accum = nl.zeros((128, out_width), dtype=nl.float32, buffer=nl.psum)

                    # convolution: go over filter kernel and input tiles
                    for fh in nl.affine_range(filter_height):
                        for fw in nl.affine_range(filter_width):
                            for ic_tile in nl.affine_range(n_tiles_c_in):
                                accum += nl.matmul(
                                    w_trans[fh, fw, oc_tile, ic_tile],
                                    x_sbuf[ic_tile, :, out_r+fh, fw:fw+out_width],
                                    transpose_x=True
                                )

                    accum = nisa.tensor_scalar(accum, np.add, bias_sbuf)

                    y_sbuf[:, out_r, :] = nl.copy(accum, dtype=X.dtype)

                nl.store(
                    X_out[b_idx, oc_tile*c_out_block:(oc_tile+1)*c_out_block, chunk_idx*chunk_size:(chunk_idx+1)*chunk_size],
                    y_sbuf
                )

    return X_out

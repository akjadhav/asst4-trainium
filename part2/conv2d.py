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
    
    # Can assume multiple of 128 to avoid using mask
    assert in_channels % 128 == 0

    # Can assume one PSUM bank can at least fit one row of the pixels
    assert nl.tile_size.gemm_moving_fmax >= out_width

    # Initialize output array
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    # Various tiling dimensions (You may want to define more of them)
    c_in_pmax = nl.tile_size.pmax
    n_tiles_c_in = in_channels // c_in_pmax
    c_out_pmax = c_in_pmax # max output 
    n_tiles_c_out = out_channels // c_out_pmax

    # prepare weights
    W = W.reshape((n_tiles_c_out, c_out_pmax, n_tiles_c_in, c_in_pmax, filter_height, filter_width))
    weight_sbuf = nl.ndarray(
            (n_tiles_c_out, nl.par_dim(c_out_pmax), n_tiles_c_in, c_in_pmax, filter_height, filter_width), 
            dtype=W.dtype,
            buffer=nl.sbuf
        )
    weight_copy = nl.ndarray(
            (filter_height, filter_width, n_tiles_c_out, n_tiles_c_in, nl.par_dim(c_out_pmax), c_in_pmax), 
            dtype=W.dtype, 
            buffer=nl.sbuf
        )
    w = nl.ndarray(
            (filter_height, filter_width, n_tiles_c_out, n_tiles_c_in, nl.par_dim(c_in_pmax), c_out_pmax),
            dtype=W.dtype,
            buffer=nl.sbuf
        )

    for c_out_tile in nl.affine_range(n_tiles_c_out):
        weight_sbuf[c_out_tile] = nl.load(W[c_out_tile])

    for c_out_tile in nl.affine_range(n_tiles_c_out):
        for c_in_tile in nl.affine_range(n_tiles_c_in):
            for i in nl.affine_range(filter_height):
                for j in nl.affine_range(filter_width):
                    weight_copy[i, j, c_out_tile, c_in_tile, :, :] = nl.copy(
                            weight_sbuf[c_out_tile, :, c_in_tile, :, i, j], dtype=W.dtype
                        )
                    w[i, j, c_out_tile, c_in_tile] = nisa.nc_transpose(weight_copy[i, j, c_out_tile, c_in_tile])

    # Define the number of output rows to process at a time
    # Start with a smaller number that's divisible by 2. Adjust as needed.
    out_chunk = 2
    n_chunks = (out_height + out_chunk - 1) // out_chunk

    for b in nl.affine_range(batch_size):
        # Process the output in chunks
        for c in range(n_chunks):
            chunk_start = c * out_chunk
            chunk_end = min((c + 1) * out_chunk, out_height)
            this_chunk_height = chunk_end - chunk_start

            # Determine how many input rows we need
            # Filter extends beyond the output rows, so we add filter_height - 1
            in_rows = this_chunk_height + filter_height - 1

            # Ensure shape elements are integers
            in_rows_int = int(in_rows)
            input_w = int(input_width)
            # Load only the required input rows
            x = nl.ndarray(
                    shape=(n_tiles_c_in, nl.par_dim(c_in_pmax), in_rows_int, input_w),
                    dtype=X.dtype, 
                    buffer=nl.sbuf,
            )
            # Here we ensure we do not go out of input bounds
            # input_load_start = chunk_start
            # input_load_end = chunk_start + in_rows_int - 1
            input_load_start = chunk_start
            input_load_end = chunk_start + in_rows_int - 1
            if input_load_end >= input_height:
                # For smaller images, ensure we don't go out of range
                input_load_end = input_height - 1
                # Adjust in_rows_int if we had to shrink
                in_rows_int = input_load_end - input_load_start + 1
                x = nl.ndarray(
                        shape=(n_tiles_c_in, nl.par_dim(c_in_pmax), in_rows_int, input_w),
                        dtype=X.dtype, 
                        buffer=nl.sbuf,
                )

            for tile_in in nl.affine_range(n_tiles_c_in):
                x[tile_in] = nl.load(
                        X[b, tile_in * c_in_pmax : (tile_in + 1) * c_in_pmax, input_load_start : input_load_end + 1, :]
                )

            # Load bias tile
            # We must incorporate bias addition as per instructions
            # Because bias is per output channel, we load it once per c_out_tile chunk
            for tile_out in nl.affine_range(n_tiles_c_out):
                output_sbuf = nl.ndarray(
                        (nl.par_dim(c_out_pmax), out_chunk, out_width),
                        dtype=X.dtype,
                        buffer=nl.sbuf
                    )
                bias_tile = nl.ndarray(
                        (nl.par_dim(c_out_pmax),), 
                        dtype=bias.dtype, 
                        buffer=nl.sbuf
                    )
                bias_tile = nl.load(bias[tile_out*c_out_pmax : (tile_out+1)*c_out_pmax])

                for out_row in nl.affine_range(this_chunk_height):
                    # out_row must not exceed the in_rows_int - filter_height + 1 range
                    # but by construction it doesn't
                    result = nl.zeros((128, out_width), dtype=nl.float32, buffer=nl.psum)

                    for i in nl.affine_range(filter_height):
                        # Ensure we don't go out of loaded input rows
                        if out_row + i >= in_rows_int:
                            break
                        for j in nl.affine_range(filter_width):
                            # Ensure indexing doesn't exceed
                            if j + out_width > input_width:
                                break
                            for tile_in in nl.affine_range(n_tiles_c_in):
                                result += nl.matmul(
                                        w[i, j, tile_out, tile_in],
                                        x[tile_in, :, out_row + i, j : j + out_width],
                                        transpose_x=True
                                )
                    # Add bias to the result
                    result = nisa.tensor_scalar(result, np.add, bias_tile)

                    output_sbuf[:, out_row, :] = nl.copy(result)

                # Store the result for this chunk of output rows
                # Handle if it's the last chunk with fewer rows
                nl.store(
                        X_out[b, tile_out * c_out_pmax : (tile_out + 1) * c_out_pmax, chunk_start : chunk_start + this_chunk_height, :],
                        value=output_sbuf[:, 0:this_chunk_height, :],
                    )

    return X_out

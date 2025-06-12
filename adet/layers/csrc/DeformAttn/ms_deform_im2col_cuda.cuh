/*!
**************************************************************************
* Deformable DETR
* Copyright (c) 2020 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 [see LICENSE for details]
**************************************************************************
* Modified from DCN (https://github.com/msracver/Deformable-ConvNets)
* Copyright (c) 2018 Microsoft
**************************************************************************
*/

#pragma once // Use pragma once for header guard

#include <cstdio>
#include <algorithm>
#include <cstring>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

//#include <THC/THCAtomics.cuh>
#include <cuda_fp16.h>
#include <type_traits> // For std::is_same
//#include <ATen/native/cuda/Loops.cuh> // Might be needed for atomicAdd on half


#define CUDA_KERNEL_LOOP(i, n)                          \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;   \
      i < (n);                                          \
      i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;
inline int GET_BLOCKS(const int N, const int num_threads)
{
  return (N + num_threads - 1) / num_threads;
}


// Bilinear interpolation helper (forward pass)
template <typename scalar_t>
__device__ __forceinline__ scalar_t ms_deform_attn_im2col_bilinear(
    const scalar_t* bottom_data,
    const int height, const int width, const int nheads, const int channels,
    const scalar_t h, const scalar_t w, const int m, const int c)
{
    const int h_low = floorf(static_cast<float>(h));
    const int w_low = floorf(static_cast<float>(w));
    const int h_high = h_low + 1;
    const int w_high = w_low + 1;

    const float h_f = static_cast<float>(h);
    const float w_f = static_cast<float>(w);
    const float lh_f = h_f - static_cast<float>(h_low);
    const float lw_f = w_f - static_cast<float>(w_low);
    const float hh_f = 1.0f - lh_f;
    const float hw_f = 1.0f - lw_f;

    const float w1_f = hh_f * hw_f;
    const float w2_f = hh_f * lw_f;
    const float w3_f = lh_f * hw_f;
    const float w4_f = lh_f * lw_f;

    const int w_stride = nheads * channels;
    const int h_stride = width * w_stride;
    const int h_low_ptr_offset = h_low * h_stride;
    const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
    const int w_low_ptr_offset = w_low * w_stride;
    const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
    const int base_ptr = m * channels + c;

    scalar_t v1 = 0, v2 = 0, v3 = 0, v4 = 0;

    if (h_low >= 0 && w_low >= 0 && h_low < height && w_low < width) {
        v1 = bottom_data[h_low_ptr_offset + w_low_ptr_offset + base_ptr];
    }
    if (h_low >= 0 && w_high < width && h_low < height) {
        v2 = bottom_data[h_low_ptr_offset + w_high_ptr_offset + base_ptr];
    }
    if (h_high < height && w_low >= 0 && w_low < width) {
        v3 = bottom_data[h_high_ptr_offset + w_low_ptr_offset + base_ptr];
    }
    if (h_high < height && w_high < width) {
        v4 = bottom_data[h_high_ptr_offset + w_high_ptr_offset + base_ptr];
    }

    const float val_f = (w1_f * static_cast<float>(v1) + w2_f * static_cast<float>(v2) +
                         w3_f * static_cast<float>(v3) + w4_f * static_cast<float>(v4));

    return static_cast<scalar_t>(val_f);
}


// Bilinear interpolation helper (backward pass - gradient accumulation)
template <typename scalar_t>
__device__ __forceinline__ void ms_deform_attn_col2im_bilinear(
    const scalar_t* bottom_data,
    const int height, const int width, const int nheads, const int channels,
    const scalar_t h, const scalar_t w, const int m, const int c,
    const scalar_t top_grad,
    const scalar_t attn_weight,
    scalar_t* grad_value,
    scalar_t* grad_sampling_loc,
    scalar_t* grad_attn_weight)
{
    const int h_low = floorf(static_cast<float>(h));
    const int w_low = floorf(static_cast<float>(w));
    const int h_high = h_low + 1;
    const int w_high = w_low + 1;

    const float h_f = static_cast<float>(h);
    const float w_f = static_cast<float>(w);
    const float lh_f = h_f - static_cast<float>(h_low);
    const float lw_f = w_f - static_cast<float>(w_low);
    const float hh_f = 1.0f - lh_f;
    const float hw_f = 1.0f - lw_f;

    const float w1_f = hh_f * hw_f;
    const float w2_f = hh_f * lw_f;
    const float w3_f = lh_f * hw_f;
    const float w4_f = lh_f * lw_f;

    const float top_grad_f = static_cast<float>(top_grad);
    const float attn_weight_f = static_cast<float>(attn_weight);
    const float top_grad_value_f = top_grad_f * attn_weight_f;

    const int w_stride = nheads * channels;
    const int h_stride = width * w_stride;
    const int h_low_ptr_offset = h_low * h_stride;
    const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
    const int w_low_ptr_offset = w_low * w_stride;
    const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
    const int base_ptr = m * channels + c;

    float grad_h_weight_f = 0.0f;
    float grad_w_weight_f = 0.0f;

    scalar_t v1 = 0, v2 = 0, v3 = 0, v4 = 0;

    if (h_low >= 0 && w_low >= 0 && h_low < height && w_low < width) {
        const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
        v1 = bottom_data[ptr1];
        // Explicit atomicAdd based on type
        if constexpr (std::is_same<scalar_t, float>::value) {
            ::atomicAdd(grad_value + ptr1, w1_f * top_grad_value_f);
        } else if constexpr (std::is_same<scalar_t, at::Half>::value) {
            __half val_to_add = __float2half(w1_f * top_grad_value_f); // Use __float2half for conversion
            unsigned short *address_as_ushort = reinterpret_cast<unsigned short *>(grad_value + ptr1);
            unsigned short old_bits = *address_as_ushort;
            unsigned short assumed_bits;
            do {
                assumed_bits = old_bits;
                old_bits = atomicCAS(address_as_ushort, assumed_bits, __half_as_ushort(__hadd(__ushort_as_half(assumed_bits), val_to_add)));
            } while (assumed_bits != old_bits);
        }
        grad_h_weight_f -= hw_f * static_cast<float>(v1);
        grad_w_weight_f -= hh_f * static_cast<float>(v1);
    }

    if (h_low >= 0 && w_high < width && h_low < height) {
        const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
        v2 = bottom_data[ptr2];
        // Explicit atomicAdd based on type
        if constexpr (std::is_same<scalar_t, float>::value) {
            ::atomicAdd(grad_value + ptr2, w2_f * top_grad_value_f);
        } else if constexpr (std::is_same<scalar_t, at::Half>::value) {
             __half val_to_add = __float2half(w2_f * top_grad_value_f);
            unsigned short *address_as_ushort = reinterpret_cast<unsigned short *>(grad_value + ptr2);
            unsigned short old_bits = *address_as_ushort;
            unsigned short assumed_bits;
            do {
                assumed_bits = old_bits;
                old_bits = atomicCAS(address_as_ushort, assumed_bits, __half_as_ushort(__hadd(__ushort_as_half(assumed_bits), val_to_add)));
            } while (assumed_bits != old_bits);
        }
        grad_h_weight_f -= lw_f * static_cast<float>(v2);
        grad_w_weight_f += hh_f * static_cast<float>(v2);
    }

     if (h_high < height && w_low >= 0 && w_low < width) {
        const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
        v3 = bottom_data[ptr3];
        // Explicit atomicAdd based on type
        if constexpr (std::is_same<scalar_t, float>::value) {
            ::atomicAdd(grad_value + ptr3, w3_f * top_grad_value_f);
        } else if constexpr (std::is_same<scalar_t, at::Half>::value) {
             __half val_to_add = __float2half(w3_f * top_grad_value_f);
            unsigned short *address_as_ushort = reinterpret_cast<unsigned short *>(grad_value + ptr3);
            unsigned short old_bits = *address_as_ushort;
            unsigned short assumed_bits;
            do {
                assumed_bits = old_bits;
                old_bits = atomicCAS(address_as_ushort, assumed_bits, __half_as_ushort(__hadd(__ushort_as_half(assumed_bits), val_to_add)));
            } while (assumed_bits != old_bits);
        }
        grad_h_weight_f += hw_f * static_cast<float>(v3);
        grad_w_weight_f -= lh_f * static_cast<float>(v3);
    }

     if (h_high < height && w_high < width) {
        const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
        v4 = bottom_data[ptr4];
        // Explicit atomicAdd based on type
        if constexpr (std::is_same<scalar_t, float>::value) {
            ::atomicAdd(grad_value + ptr4, w4_f * top_grad_value_f);
        } else if constexpr (std::is_same<scalar_t, at::Half>::value) {
             __half val_to_add = __float2half(w4_f * top_grad_value_f);
            unsigned short *address_as_ushort = reinterpret_cast<unsigned short *>(grad_value + ptr4);
            unsigned short old_bits = *address_as_ushort;
            unsigned short assumed_bits;
            do {
                assumed_bits = old_bits;
                old_bits = atomicCAS(address_as_ushort, assumed_bits, __half_as_ushort(__hadd(__ushort_as_half(assumed_bits), val_to_add)));
            } while (assumed_bits != old_bits);
        }
        grad_h_weight_f += lw_f * static_cast<float>(v4);
        grad_w_weight_f += lh_f * static_cast<float>(v4);
    }

    const float val_f = (w1_f * static_cast<float>(v1) + w2_f * static_cast<float>(v2) +
                         w3_f * static_cast<float>(v3) + w4_f * static_cast<float>(v4));

    *grad_attn_weight = static_cast<scalar_t>(top_grad_f * val_f);
    *grad_sampling_loc = static_cast<scalar_t>(static_cast<float>(width) * grad_w_weight_f * top_grad_value_f);
    *(grad_sampling_loc + 1) = static_cast<scalar_t>(static_cast<float>(height) * grad_h_weight_f * top_grad_value_f);
}

template <typename scalar_t>
__device__ void ms_deform_attn_col2im_bilinear_gm(const scalar_t* bottom_data,
                                                  const int &height, const int &width, const int &nheads, const int &channels,
                                                  const scalar_t &h, const scalar_t &w, const int &m, const int &c,
                                                  const scalar_t &top_grad,
                                                  const scalar_t &attn_weight,
                                                  scalar_t* grad_value,
                                                  scalar_t* grad_sampling_loc,
                                                  scalar_t* grad_attn_weight)
{
    // floor/ceil fixes
    const int h_low = floorf(static_cast<float>(h));
    const int w_low = floorf(static_cast<float>(w));
    const int h_high = h_low + 1;
    const int w_high = w_low + 1;

    // float calculations
    const float h_f = static_cast<float>(h);
    const float w_f = static_cast<float>(w);
    const float lh_f = h_f - h_low;
    const float lw_f = w_f - w_low;
    const float hh_f = 1.0f - lh_f;
    const float hw_f = 1.0f - lw_f;

    const float w1_f = hh_f * hw_f;
    const float w2_f = hh_f * lw_f;
    const float w3_f = lh_f * hw_f;
    const float w4_f = lh_f * lw_f;
    const float top_grad_f = static_cast<float>(top_grad);
    const float attn_weight_f = static_cast<float>(attn_weight);
    const float top_grad_value_f = top_grad_f * attn_weight_f;

    // pointer calculations
    const int w_stride = nheads * channels;
    const int h_stride = width * w_stride;
    const int h_low_ptr_offset = h_low * h_stride;
    const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
    const int w_low_ptr_offset = w_low * w_stride;
    const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
    const int base_ptr = m * channels + c;

    float grad_h_weight_f = 0.0f;
    float grad_w_weight_f = 0.0f;
    scalar_t v1 = 0, v2 = 0, v3 = 0, v4 = 0;

    // --- Gradient accumulation for grad_value (atomic) ---
    if (h_low >= 0 && w_low >= 0 && h_low < height && w_low < width) {
        const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
        v1 = bottom_data[ptr1];
        // Explicit atomicAdd based on type
        if constexpr (std::is_same<scalar_t, float>::value) {
             ::atomicAdd(grad_value + ptr1, w1_f * top_grad_value_f);
        } else if constexpr (std::is_same<scalar_t, at::Half>::value) {
            unsigned short *address_as_ushort = reinterpret_cast<unsigned short *>(grad_value + ptr1);
            unsigned short old_bits = *address_as_ushort;
            unsigned short assumed_bits;
             __half value_to_add_h = __float2half(w1_f * top_grad_value_f); // Use float2half conversion
            do {
                assumed_bits = old_bits;
                old_bits = atomicCAS(address_as_ushort, assumed_bits, __half_as_ushort(__hadd(__ushort_as_half(assumed_bits), value_to_add_h)));
            } while (assumed_bits != old_bits);
        }
        grad_h_weight_f -= hw_f * static_cast<float>(v1);
        grad_w_weight_f -= hh_f * static_cast<float>(v1);
    }
    // ... similar explicit atomicAdd logic for ptr2, ptr3, ptr4 ...
    if (h_low >= 0 && w_high < width && h_low < height) {
        const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
        v2 = bottom_data[ptr2];
        if constexpr (std::is_same<scalar_t, float>::value) { ::atomicAdd(grad_value + ptr2, w2_f * top_grad_value_f); }
        else if constexpr (std::is_same<scalar_t, at::Half>::value) { /* atomicCAS logic for ptr2 */
             unsigned short *address_as_ushort = reinterpret_cast<unsigned short *>(grad_value + ptr2);
             unsigned short old_bits = *address_as_ushort; unsigned short assumed_bits;
             __half value_to_add_h = __float2half(w2_f * top_grad_value_f);
             do { assumed_bits = old_bits; old_bits = atomicCAS(address_as_ushort, assumed_bits, __half_as_ushort(__hadd(__ushort_as_half(assumed_bits), value_to_add_h))); } while (assumed_bits != old_bits);
        }
        grad_h_weight_f -= lw_f * static_cast<float>(v2); grad_w_weight_f += hh_f * static_cast<float>(v2);
    }
    if (h_high < height && w_low >= 0 && w_low < width) {
        const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
        v3 = bottom_data[ptr3];
        if constexpr (std::is_same<scalar_t, float>::value) { ::atomicAdd(grad_value + ptr3, w3_f * top_grad_value_f); }
        else if constexpr (std::is_same<scalar_t, at::Half>::value) { /* atomicCAS logic for ptr3 */
             unsigned short *address_as_ushort = reinterpret_cast<unsigned short *>(grad_value + ptr3);
             unsigned short old_bits = *address_as_ushort; unsigned short assumed_bits;
             __half value_to_add_h = __float2half(w3_f * top_grad_value_f);
             do { assumed_bits = old_bits; old_bits = atomicCAS(address_as_ushort, assumed_bits, __half_as_ushort(__hadd(__ushort_as_half(assumed_bits), value_to_add_h))); } while (assumed_bits != old_bits);
        }
        grad_h_weight_f += hw_f * static_cast<float>(v3); grad_w_weight_f -= lh_f * static_cast<float>(v3);
    }
    if (h_high < height && w_high < width) {
        const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
        v4 = bottom_data[ptr4];
        if constexpr (std::is_same<scalar_t, float>::value) { ::atomicAdd(grad_value + ptr4, w4_f * top_grad_value_f); }
        else if constexpr (std::is_same<scalar_t, at::Half>::value) { /* atomicCAS logic for ptr4 */
            unsigned short *address_as_ushort = reinterpret_cast<unsigned short *>(grad_value + ptr4);
            unsigned short old_bits = *address_as_ushort; unsigned short assumed_bits;
            __half value_to_add_h = __float2half(w4_f * top_grad_value_f);
            do { assumed_bits = old_bits; old_bits = atomicCAS(address_as_ushort, assumed_bits, __half_as_ushort(__hadd(__ushort_as_half(assumed_bits), value_to_add_h))); } while (assumed_bits != old_bits);
        }
        grad_h_weight_f += lw_f * static_cast<float>(v4); grad_w_weight_f += lh_f * static_cast<float>(v4);
    }


    // --- Final gradient calculations (atomic for _gm version) ---
    const float val_f = (w1_f * static_cast<float>(v1) + w2_f * static_cast<float>(v2) +
                         w3_f * static_cast<float>(v3) + w4_f * static_cast<float>(v4));

    // Explicit atomicAdd for sampling_loc and attn_weight
    if constexpr (std::is_same<scalar_t, float>::value) {
        ::atomicAdd(grad_attn_weight, top_grad_f * val_f);
        ::atomicAdd(grad_sampling_loc, static_cast<float>(width) * grad_w_weight_f * top_grad_value_f);
        ::atomicAdd(grad_sampling_loc + 1, static_cast<float>(height) * grad_h_weight_f * top_grad_value_f);
    } else if constexpr (std::is_same<scalar_t, at::Half>::value) {
        // atomicCAS loop for grad_attn_weight
        unsigned short *aw_addr = reinterpret_cast<unsigned short *>(grad_attn_weight);
        unsigned short aw_old_bits = *aw_addr; unsigned short aw_assumed_bits;
        __half aw_val_to_add_h = __float2half(top_grad_f * val_f);
        do { aw_assumed_bits = aw_old_bits; aw_old_bits = atomicCAS(aw_addr, aw_assumed_bits, __half_as_ushort(__hadd(__ushort_as_half(aw_assumed_bits), aw_val_to_add_h))); } while (aw_assumed_bits != aw_old_bits);

        // atomicCAS loop for grad_sampling_loc (x)
        unsigned short *slx_addr = reinterpret_cast<unsigned short *>(grad_sampling_loc);
        unsigned short slx_old_bits = *slx_addr; unsigned short slx_assumed_bits;
        __half slx_val_to_add_h = __float2half(static_cast<float>(width) * grad_w_weight_f * top_grad_value_f);
        do { slx_assumed_bits = slx_old_bits; slx_old_bits = atomicCAS(slx_addr, slx_assumed_bits, __half_as_ushort(__hadd(__ushort_as_half(slx_assumed_bits), slx_val_to_add_h))); } while (slx_assumed_bits != slx_old_bits);

        // atomicCAS loop for grad_sampling_loc (y)
        unsigned short *sly_addr = reinterpret_cast<unsigned short *>(grad_sampling_loc + 1);
        unsigned short sly_old_bits = *sly_addr; unsigned short sly_assumed_bits;
        __half sly_val_to_add_h = __float2half(static_cast<float>(height) * grad_h_weight_f * top_grad_value_f);
        do { sly_assumed_bits = sly_old_bits; sly_old_bits = atomicCAS(sly_addr, sly_assumed_bits, __half_as_ushort(__hadd(__ushort_as_half(sly_assumed_bits), sly_val_to_add_h))); } while (sly_assumed_bits != sly_old_bits);
    }
}


template <typename scalar_t>
__global__ void ms_deformable_im2col_gpu_kernel(const int n,
                                                const scalar_t *data_value, 
                                                const int64_t *data_spatial_shapes,
                                                const int64_t *data_level_start_index, 
                                                const scalar_t *data_sampling_loc,
                                                const scalar_t *data_attn_weight,
                                                const int batch_size, 
                                                const int spatial_size, 
                                                const int num_heads,
                                                const int channels, 
                                                const int num_levels,
                                                const int num_query,
                                                const int num_point,
                                                scalar_t *data_col)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp; 
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    const int q_col = _temp % num_query;
    _temp /= num_query;
    const int b_col = _temp;

    scalar_t *data_col_ptr = data_col + index;
    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;
    scalar_t col = 0;
    
    for (int l_col=0; l_col < num_levels; ++l_col)
    {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const scalar_t *data_value_ptr = data_value + (data_value_ptr_init_offset + level_start_id * qid_stride);
      for (int p_col=0; p_col < num_point; ++p_col)
      {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;

        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w)
        {
          col += ms_deform_attn_im2col_bilinear(data_value_ptr, spatial_h, spatial_w, num_heads, channels, h_im, w_im, m_col, c_col) * weight;
        }

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
      }
    }
    *data_col_ptr = col;
  }
}

template <typename scalar_t, unsigned int blockSize>
__global__ void ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1(const int n,
                                                const scalar_t *grad_col,
                                                const scalar_t *data_value,
                                                const int64_t *data_spatial_shapes,
                                                const int64_t *data_level_start_index, 
                                                const scalar_t *data_sampling_loc,
                                                const scalar_t *data_attn_weight,
                                                const int batch_size, 
                                                const int spatial_size, 
                                                const int num_heads,
                                                const int channels, 
                                                const int num_levels,
                                                const int num_query,
                                                const int num_point,
                                                scalar_t *grad_value,
                                                scalar_t *grad_sampling_loc,
                                                scalar_t *grad_attn_weight)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    __shared__ scalar_t cache_grad_sampling_loc[blockSize * 2];
    __shared__ scalar_t cache_grad_attn_weight[blockSize];
    unsigned int tid = threadIdx.x;
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp; 
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    const int q_col = _temp % num_query;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int grad_sampling_ptr = data_weight_ptr;
    grad_sampling_loc += grad_sampling_ptr << 1;
    grad_attn_weight += grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 2;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col=0; l_col < num_levels; ++l_col)
    {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int value_ptr_offset = data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col=0; p_col < num_point; ++p_col)
      {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        *(cache_grad_sampling_loc+(threadIdx.x << 1)) = 0;
        *(cache_grad_sampling_loc+((threadIdx.x << 1) + 1)) = 0;
        *(cache_grad_attn_weight+threadIdx.x)=0;
        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w)
        {
          ms_deform_attn_col2im_bilinear(
            data_value_ptr, spatial_h, spatial_w, num_heads, channels, h_im, w_im, m_col, c_col,
            top_grad, weight, grad_value_ptr, 
            cache_grad_sampling_loc+(threadIdx.x << 1), cache_grad_attn_weight+threadIdx.x);
        }
        
        __syncthreads();
        if (tid == 0)
        {
          scalar_t _grad_w=cache_grad_sampling_loc[0], _grad_h=cache_grad_sampling_loc[1], _grad_a=cache_grad_attn_weight[0];
          int sid=2;
          for (unsigned int tid = 1; tid < blockSize; ++tid)
          {
            _grad_w += cache_grad_sampling_loc[sid];
            _grad_h += cache_grad_sampling_loc[sid + 1];
            _grad_a += cache_grad_attn_weight[tid];
            sid += 2;
          }
          
          
          *grad_sampling_loc = _grad_w;
          *(grad_sampling_loc + 1) = _grad_h;
          *grad_attn_weight = _grad_a;
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
        grad_attn_weight += grad_weight_stride;
        grad_sampling_loc += grad_loc_stride;
      }
    }
  }
}


template <typename scalar_t, unsigned int blockSize>
__global__ void ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2(const int n,
                                                const scalar_t *grad_col,
                                                const scalar_t *data_value,
                                                const int64_t *data_spatial_shapes,
                                                const int64_t *data_level_start_index, 
                                                const scalar_t *data_sampling_loc,
                                                const scalar_t *data_attn_weight,
                                                const int batch_size, 
                                                const int spatial_size, 
                                                const int num_heads,
                                                const int channels, 
                                                const int num_levels,
                                                const int num_query,
                                                const int num_point,
                                                scalar_t *grad_value,
                                                scalar_t *grad_sampling_loc,
                                                scalar_t *grad_attn_weight)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    __shared__ scalar_t cache_grad_sampling_loc[blockSize * 2];
    __shared__ scalar_t cache_grad_attn_weight[blockSize];
    unsigned int tid = threadIdx.x;
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp; 
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    const int q_col = _temp % num_query;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int grad_sampling_ptr = data_weight_ptr;
    grad_sampling_loc += grad_sampling_ptr << 1;
    grad_attn_weight += grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 2;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col=0; l_col < num_levels; ++l_col)
    {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int value_ptr_offset = data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col=0; p_col < num_point; ++p_col)
      {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        *(cache_grad_sampling_loc+(threadIdx.x << 1)) = 0;
        *(cache_grad_sampling_loc+((threadIdx.x << 1) + 1)) = 0;
        *(cache_grad_attn_weight+threadIdx.x)=0;
        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w)
        {
          ms_deform_attn_col2im_bilinear(
            data_value_ptr, spatial_h, spatial_w, num_heads, channels, h_im, w_im, m_col, c_col,
            top_grad, weight, grad_value_ptr, 
            cache_grad_sampling_loc+(threadIdx.x << 1), cache_grad_attn_weight+threadIdx.x);
        }
        
        __syncthreads();

        for (unsigned int s=blockSize/2; s>0; s>>=1)
        {
          if (tid < s) {
            const unsigned int xid1 = tid << 1;
            const unsigned int xid2 = (tid + s) << 1;
            cache_grad_attn_weight[tid] += cache_grad_attn_weight[tid + s];
            cache_grad_sampling_loc[xid1] += cache_grad_sampling_loc[xid2];
            cache_grad_sampling_loc[xid1 + 1] += cache_grad_sampling_loc[xid2 + 1];
          }
          __syncthreads();
        }

        if (tid == 0)
        { 
          *grad_sampling_loc = cache_grad_sampling_loc[0];
          *(grad_sampling_loc + 1) = cache_grad_sampling_loc[1];
          *grad_attn_weight = cache_grad_attn_weight[0];
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
        grad_attn_weight += grad_weight_stride;
        grad_sampling_loc += grad_loc_stride;
      }
    }
  }
}


template <typename scalar_t>
__global__ void ms_deformable_col2im_gpu_kernel_shm_reduce_v1(const int n,
                                                const scalar_t *grad_col,
                                                const scalar_t *data_value,
                                                const int64_t *data_spatial_shapes,
                                                const int64_t *data_level_start_index, 
                                                const scalar_t *data_sampling_loc,
                                                const scalar_t *data_attn_weight,
                                                const int batch_size, 
                                                const int spatial_size, 
                                                const int num_heads,
                                                const int channels, 
                                                const int num_levels,
                                                const int num_query,
                                                const int num_point,
                                                scalar_t *grad_value,
                                                scalar_t *grad_sampling_loc,
                                                scalar_t *grad_attn_weight)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    extern __shared__ int _s[];
    scalar_t* cache_grad_sampling_loc = (scalar_t*)_s;
    scalar_t* cache_grad_attn_weight = cache_grad_sampling_loc + 2 * blockDim.x;
    unsigned int tid = threadIdx.x;
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp; 
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    const int q_col = _temp % num_query;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int grad_sampling_ptr = data_weight_ptr;
    grad_sampling_loc += grad_sampling_ptr << 1;
    grad_attn_weight += grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 2;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col=0; l_col < num_levels; ++l_col)
    {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int value_ptr_offset = data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col=0; p_col < num_point; ++p_col)
      {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        *(cache_grad_sampling_loc+(threadIdx.x << 1)) = 0;
        *(cache_grad_sampling_loc+((threadIdx.x << 1) + 1)) = 0;
        *(cache_grad_attn_weight+threadIdx.x)=0;
        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w)
        {
          ms_deform_attn_col2im_bilinear(
            data_value_ptr, spatial_h, spatial_w, num_heads, channels, h_im, w_im, m_col, c_col,
            top_grad, weight, grad_value_ptr, 
            cache_grad_sampling_loc+(threadIdx.x << 1), cache_grad_attn_weight+threadIdx.x);
        }
        
        __syncthreads();
        if (tid == 0)
        {
          scalar_t _grad_w=cache_grad_sampling_loc[0], _grad_h=cache_grad_sampling_loc[1], _grad_a=cache_grad_attn_weight[0];
          int sid=2;
          for (unsigned int tid = 1; tid < blockDim.x; ++tid)
          {
            _grad_w += cache_grad_sampling_loc[sid];
            _grad_h += cache_grad_sampling_loc[sid + 1];
            _grad_a += cache_grad_attn_weight[tid];
            sid += 2;
          }
          
          
          *grad_sampling_loc = _grad_w;
          *(grad_sampling_loc + 1) = _grad_h;
          *grad_attn_weight = _grad_a;
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
        grad_attn_weight += grad_weight_stride;
        grad_sampling_loc += grad_loc_stride;
      }
    }
  }
}

template <typename scalar_t>
__global__ void ms_deformable_col2im_gpu_kernel_shm_reduce_v2(const int n,
                                                const scalar_t *grad_col,
                                                const scalar_t *data_value,
                                                const int64_t *data_spatial_shapes,
                                                const int64_t *data_level_start_index, 
                                                const scalar_t *data_sampling_loc,
                                                const scalar_t *data_attn_weight,
                                                const int batch_size, 
                                                const int spatial_size, 
                                                const int num_heads,
                                                const int channels, 
                                                const int num_levels,
                                                const int num_query,
                                                const int num_point,
                                                scalar_t *grad_value,
                                                scalar_t *grad_sampling_loc,
                                                scalar_t *grad_attn_weight)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    extern __shared__ int _s[];
    scalar_t* cache_grad_sampling_loc = (scalar_t*)_s;
    scalar_t* cache_grad_attn_weight = cache_grad_sampling_loc + 2 * blockDim.x;
    unsigned int tid = threadIdx.x;
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp; 
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    const int q_col = _temp % num_query;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int grad_sampling_ptr = data_weight_ptr;
    grad_sampling_loc += grad_sampling_ptr << 1;
    grad_attn_weight += grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 2;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col=0; l_col < num_levels; ++l_col)
    {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int value_ptr_offset = data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col=0; p_col < num_point; ++p_col)
      {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        *(cache_grad_sampling_loc+(threadIdx.x << 1)) = 0;
        *(cache_grad_sampling_loc+((threadIdx.x << 1) + 1)) = 0;
        *(cache_grad_attn_weight+threadIdx.x)=0;
        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w)
        {
          ms_deform_attn_col2im_bilinear(
            data_value_ptr, spatial_h, spatial_w, num_heads, channels, h_im, w_im, m_col, c_col,
            top_grad, weight, grad_value_ptr, 
            cache_grad_sampling_loc+(threadIdx.x << 1), cache_grad_attn_weight+threadIdx.x);
        }
        
        __syncthreads();

        for (unsigned int s=blockDim.x/2, spre=blockDim.x; s>0; s>>=1, spre>>=1)
        {
          if (tid < s) {
            const unsigned int xid1 = tid << 1;
            const unsigned int xid2 = (tid + s) << 1;
            cache_grad_attn_weight[tid] += cache_grad_attn_weight[tid + s];
            cache_grad_sampling_loc[xid1] += cache_grad_sampling_loc[xid2];
            cache_grad_sampling_loc[xid1 + 1] += cache_grad_sampling_loc[xid2 + 1];
            if (tid + (s << 1) < spre)
            {
              cache_grad_attn_weight[tid] += cache_grad_attn_weight[tid + (s << 1)];
              cache_grad_sampling_loc[xid1] += cache_grad_sampling_loc[xid2 + (s << 1)];
              cache_grad_sampling_loc[xid1 + 1] += cache_grad_sampling_loc[xid2 + 1 + (s << 1)];
            } 
          }
          __syncthreads();
        }

        if (tid == 0)
        {
          *grad_sampling_loc = cache_grad_sampling_loc[0];
          *(grad_sampling_loc + 1) = cache_grad_sampling_loc[1];
          *grad_attn_weight = cache_grad_attn_weight[0];
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
        grad_attn_weight += grad_weight_stride;
        grad_sampling_loc += grad_loc_stride;
      }
    }
  }
}

template <typename scalar_t>
__global__ void ms_deformable_col2im_gpu_kernel_shm_reduce_v2_multi_blocks(const int n,
                                                const scalar_t *grad_col,
                                                const scalar_t *data_value,
                                                const int64_t *data_spatial_shapes,
                                                const int64_t *data_level_start_index, 
                                                const scalar_t *data_sampling_loc,
                                                const scalar_t *data_attn_weight,
                                                const int batch_size, 
                                                const int spatial_size, 
                                                const int num_heads,
                                                const int channels, 
                                                const int num_levels,
                                                const int num_query,
                                                const int num_point,
                                                scalar_t *grad_value,
                                                scalar_t *grad_sampling_loc,
                                                scalar_t *grad_attn_weight)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    extern __shared__ int _s[];
    scalar_t* cache_grad_sampling_loc = (scalar_t*)_s;
    scalar_t* cache_grad_attn_weight = cache_grad_sampling_loc + 2 * blockDim.x;
    unsigned int tid = threadIdx.x;
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp; 
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    const int q_col = _temp % num_query;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int grad_sampling_ptr = data_weight_ptr;
    grad_sampling_loc += grad_sampling_ptr << 1;
    grad_attn_weight += grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 2;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col=0; l_col < num_levels; ++l_col)
    {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int value_ptr_offset = data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col=0; p_col < num_point; ++p_col)
      {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        *(cache_grad_sampling_loc+(threadIdx.x << 1)) = 0;
        *(cache_grad_sampling_loc+((threadIdx.x << 1) + 1)) = 0;
        *(cache_grad_attn_weight+threadIdx.x)=0;
        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w)
        {
          ms_deform_attn_col2im_bilinear(
            data_value_ptr, spatial_h, spatial_w, num_heads, channels, h_im, w_im, m_col, c_col,
            top_grad, weight, grad_value_ptr, 
            cache_grad_sampling_loc+(threadIdx.x << 1), cache_grad_attn_weight+threadIdx.x);
        }
        
        __syncthreads();

        for (unsigned int s=blockDim.x/2, spre=blockDim.x; s>0; s>>=1, spre>>=1)
        {
          if (tid < s) {
            const unsigned int xid1 = tid << 1;
            const unsigned int xid2 = (tid + s) << 1;
            cache_grad_attn_weight[tid] += cache_grad_attn_weight[tid + s];
            cache_grad_sampling_loc[xid1] += cache_grad_sampling_loc[xid2];
            cache_grad_sampling_loc[xid1 + 1] += cache_grad_sampling_loc[xid2 + 1];
            if (tid + (s << 1) < spre)
            {
              cache_grad_attn_weight[tid] += cache_grad_attn_weight[tid + (s << 1)];
              cache_grad_sampling_loc[xid1] += cache_grad_sampling_loc[xid2 + (s << 1)];
              cache_grad_sampling_loc[xid1 + 1] += cache_grad_sampling_loc[xid2 + 1 + (s << 1)];
            }
          }
          __syncthreads();
        }

        if (tid == 0)
        {
          // Get the reduced values from shared memory
          scalar_t reduced_grad_loc_w = cache_grad_sampling_loc[0];
          scalar_t reduced_grad_loc_h = cache_grad_sampling_loc[1];
          scalar_t reduced_grad_attn_w = cache_grad_attn_weight[0];

          // --- Explicit atomicAdd based on type ---
          if constexpr (std::is_same<scalar_t, float>::value) {
            ::atomicAdd(grad_sampling_loc, reduced_grad_loc_w);
            ::atomicAdd(grad_sampling_loc + 1, reduced_grad_loc_h);
            ::atomicAdd(grad_attn_weight, reduced_grad_attn_w);
          }
          else if constexpr (std::is_same<scalar_t, at::Half>::value) {
            // atomicCAS for grad_sampling_loc (x)
            unsigned short *slx_addr = reinterpret_cast<unsigned short *>(grad_sampling_loc);
            unsigned short slx_old_bits = *slx_addr; unsigned short slx_assumed_bits;
            __half slx_val_to_add_h = static_cast<__half>(reduced_grad_loc_w); // Cast reduced value
            do { slx_assumed_bits = slx_old_bits; slx_old_bits = atomicCAS(slx_addr, slx_assumed_bits, __half_as_ushort(__hadd(__ushort_as_half(slx_assumed_bits), slx_val_to_add_h))); } while (slx_assumed_bits != slx_old_bits);

            // atomicCAS for grad_sampling_loc (y)
            unsigned short *sly_addr = reinterpret_cast<unsigned short *>(grad_sampling_loc + 1);
            unsigned short sly_old_bits = *sly_addr; unsigned short sly_assumed_bits;
            __half sly_val_to_add_h = static_cast<__half>(reduced_grad_loc_h); // Cast reduced value
            do { sly_assumed_bits = sly_old_bits; sly_old_bits = atomicCAS(sly_addr, sly_assumed_bits, __half_as_ushort(__hadd(__ushort_as_half(sly_assumed_bits), sly_val_to_add_h))); } while (sly_assumed_bits != sly_old_bits);

            // atomicCAS for grad_attn_weight
            unsigned short *aw_addr = reinterpret_cast<unsigned short *>(grad_attn_weight);
            unsigned short aw_old_bits = *aw_addr; unsigned short aw_assumed_bits;
            __half aw_val_to_add_h = static_cast<__half>(reduced_grad_attn_w); // Cast reduced value
            do { aw_assumed_bits = aw_old_bits; aw_old_bits = atomicCAS(aw_addr, aw_assumed_bits, __half_as_ushort(__hadd(__ushort_as_half(aw_assumed_bits), aw_val_to_add_h))); } while (aw_assumed_bits != aw_old_bits);
        }
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
        grad_attn_weight += grad_weight_stride;
        grad_sampling_loc += grad_loc_stride;
      }
    }
  }
}


template <typename scalar_t>
__global__ void ms_deformable_col2im_gpu_kernel_gm(const int n,
                                                const scalar_t *grad_col,
                                                const scalar_t *data_value,
                                                const int64_t *data_spatial_shapes,
                                                const int64_t *data_level_start_index, 
                                                const scalar_t *data_sampling_loc,
                                                const scalar_t *data_attn_weight,
                                                const int batch_size, 
                                                const int spatial_size, 
                                                const int num_heads,
                                                const int channels, 
                                                const int num_levels,
                                                const int num_query,
                                                const int num_point,
                                                scalar_t *grad_value,
                                                scalar_t *grad_sampling_loc,
                                                scalar_t *grad_attn_weight)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp; 
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    const int q_col = _temp % num_query;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int grad_sampling_ptr = data_weight_ptr;
    grad_sampling_loc += grad_sampling_ptr << 1;
    grad_attn_weight += grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 2;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col=0; l_col < num_levels; ++l_col)
    {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int value_ptr_offset = data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col=0; p_col < num_point; ++p_col)
      {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w)
        {
          ms_deform_attn_col2im_bilinear_gm(
            data_value_ptr, spatial_h, spatial_w, num_heads, channels, h_im, w_im, m_col, c_col,
            top_grad, weight, grad_value_ptr, 
            grad_sampling_loc, grad_attn_weight);
        }
        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
        grad_attn_weight += grad_weight_stride;
        grad_sampling_loc += grad_loc_stride;
      }
    }
  }
}


template <typename scalar_t>
void ms_deformable_im2col_cuda(cudaStream_t stream,
                              const scalar_t* data_value,
                              const int64_t* data_spatial_shapes, 
                              const int64_t* data_level_start_index, 
                              const scalar_t* data_sampling_loc,
                              const scalar_t* data_attn_weight,
                              const int batch_size,
                              const int spatial_size, 
                              const int num_heads, 
                              const int channels, 
                              const int num_levels, 
                              const int num_query,
                              const int num_point,
                              scalar_t* data_col)
{
  const int num_kernels = batch_size * num_query * num_heads * channels;
  const int num_actual_kernels = batch_size * num_query * num_heads * channels;
  const int num_threads = CUDA_NUM_THREADS;
  ms_deformable_im2col_gpu_kernel<scalar_t>
      <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
          0, stream>>>(
      num_kernels, data_value, data_spatial_shapes, data_level_start_index, data_sampling_loc, data_attn_weight, 
      batch_size, spatial_size, num_heads, channels, num_levels, num_query, num_point, data_col);
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in ms_deformable_im2col_cuda: %s\n", cudaGetErrorString(err));
  }

}

template <typename scalar_t>
void ms_deformable_col2im_cuda(cudaStream_t stream,
                              const scalar_t* grad_col,
                              const scalar_t* data_value,
                              const int64_t * data_spatial_shapes,
                              const int64_t * data_level_start_index,
                              const scalar_t * data_sampling_loc,
                              const scalar_t * data_attn_weight,
                              const int batch_size, 
                              const int spatial_size, 
                              const int num_heads,
                              const int channels, 
                              const int num_levels,
                              const int num_query,
                              const int num_point, 
                              scalar_t* grad_value,
                              scalar_t* grad_sampling_loc,
                              scalar_t* grad_attn_weight)
{
  const int num_threads = (channels > CUDA_NUM_THREADS)?CUDA_NUM_THREADS:channels;
  const int num_kernels = batch_size * num_query * num_heads * channels;
  const int num_actual_kernels = batch_size * num_query * num_heads * channels;
  if (channels > 1024)
  {
    if ((channels & 1023) == 0)
    {
      ms_deformable_col2im_gpu_kernel_shm_reduce_v2_multi_blocks<scalar_t>
          <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
              num_threads*3*sizeof(scalar_t), stream>>>(
                        num_kernels, 
                        grad_col,
                        data_value,
                        data_spatial_shapes,
                        data_level_start_index, 
                        data_sampling_loc,
                        data_attn_weight,
                        batch_size, 
                        spatial_size, 
                        num_heads,
                        channels, 
                        num_levels,
                        num_query,
                        num_point,
                        grad_value,
                        grad_sampling_loc,
                        grad_attn_weight);
    }
    else
    {
      ms_deformable_col2im_gpu_kernel_gm<scalar_t>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels, 
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index, 
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size, 
                      spatial_size, 
                      num_heads,
                      channels, 
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_sampling_loc,
                      grad_attn_weight);
    }
  }
  else{
    switch(channels)
    {
      case 1:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 1>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels, 
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index, 
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size, 
                      spatial_size, 
                      num_heads,
                      channels, 
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_sampling_loc,
                      grad_attn_weight);
        break;
      case 2:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 2>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels, 
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index, 
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size, 
                      spatial_size, 
                      num_heads,
                      channels, 
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_sampling_loc,
                      grad_attn_weight);
        break;
      case 4:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 4>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels, 
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index, 
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size, 
                      spatial_size, 
                      num_heads,
                      channels, 
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_sampling_loc,
                      grad_attn_weight);
        break;
      case 8:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 8>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels, 
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index, 
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size, 
                      spatial_size, 
                      num_heads,
                      channels, 
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_sampling_loc,
                      grad_attn_weight);
        break;
      case 16:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 16>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels, 
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index, 
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size, 
                      spatial_size, 
                      num_heads,
                      channels, 
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_sampling_loc,
                      grad_attn_weight);
        break;
      case 32:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 32>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels, 
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index, 
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size, 
                      spatial_size, 
                      num_heads,
                      channels, 
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_sampling_loc,
                      grad_attn_weight);
        break;
      case 64:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 64>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels, 
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index, 
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size, 
                      spatial_size, 
                      num_heads,
                      channels, 
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_sampling_loc,
                      grad_attn_weight);
        break;
      case 128:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 128>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels, 
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index, 
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size, 
                      spatial_size, 
                      num_heads,
                      channels, 
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_sampling_loc,
                      grad_attn_weight);
        break;
      case 256:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 256>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels, 
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index, 
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size, 
                      spatial_size, 
                      num_heads,
                      channels, 
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_sampling_loc,
                      grad_attn_weight);
        break;
      case 512:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 512>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels, 
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index, 
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size, 
                      spatial_size, 
                      num_heads,
                      channels, 
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_sampling_loc,
                      grad_attn_weight);
        break;
      case 1024:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 1024>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels, 
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index, 
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size, 
                      spatial_size, 
                      num_heads,
                      channels, 
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_sampling_loc,
                      grad_attn_weight);
        break;
      default:
        if (channels < 64)
        {
          ms_deformable_col2im_gpu_kernel_shm_reduce_v1<scalar_t>
          <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
              num_threads*3*sizeof(scalar_t), stream>>>(
                        num_kernels, 
                        grad_col,
                        data_value,
                        data_spatial_shapes,
                        data_level_start_index, 
                        data_sampling_loc,
                        data_attn_weight,
                        batch_size, 
                        spatial_size, 
                        num_heads,
                        channels, 
                        num_levels,
                        num_query,
                        num_point,
                        grad_value,
                        grad_sampling_loc,
                        grad_attn_weight);
        }
        else
        {
          ms_deformable_col2im_gpu_kernel_shm_reduce_v2<scalar_t>
          <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
              num_threads*3*sizeof(scalar_t), stream>>>(
                        num_kernels, 
                        grad_col,
                        data_value,
                        data_spatial_shapes,
                        data_level_start_index, 
                        data_sampling_loc,
                        data_attn_weight,
                        batch_size, 
                        spatial_size, 
                        num_heads,
                        channels, 
                        num_levels,
                        num_query,
                        num_point,
                        grad_value,
                        grad_sampling_loc,
                        grad_attn_weight);
        }
    }
  }
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in ms_deformable_col2im_cuda: %s\n", cudaGetErrorString(err));
  }

}
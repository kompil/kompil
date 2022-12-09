#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <THC/THCAtomics.cuh>

#include <vector>

#define MAX_THREADS 256
#define BATCH_PER_ITER 16
#define BATCH_PER_ITER_FORWARD BATCH_PER_ITER
#define BATCH_PER_ITER_BACKWARD BATCH_PER_ITER

namespace kernels::adjacent_1d {

__device__ inline void get_item(long idx, int s, int* c, int* x)
{
    *x = idx % s;
    *c = idx / s;
}

template <typename T>
__device__ inline T round_div_int(T val, T div)
{
    T output = (val + div / 2) / div;
    return output;
}

template <typename scalar_t>
__global__ void forward(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> batch_in, // {N, in_c, in_s}
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> weights, // {in_c, ker, out_c, out_s}
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> bias, // {out_c, out_s}
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> output, // {N, out_c, out_s}
    const int batch_size,
    const int in_c,
    const int in_s,
    const int ker,
    const int out_c,
    const int out_s)
{
    const long out_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_idx >= out_c * out_s) {
        return;
    }

    int out_ch, out_x;
    get_item(out_idx, out_s, &out_ch, &out_x);

    const int in_x = round_div_int<long>(static_cast<long>(out_x) * (in_s - ker), out_s - 1);
    const int max_x = std::min(ker, in_s - in_x);

    const scalar_t curr_bias = bias[out_ch][out_x];

    scalar_t b_curr_acc[BATCH_PER_ITER_FORWARD];
    scalar_t b_input[BATCH_PER_ITER_FORWARD];

    for (int batch = 0; batch < batch_size; batch += BATCH_PER_ITER_FORWARD) {

        int id_for_switch = std::min(batch_size - batch, BATCH_PER_ITER_FORWARD);

        #pragma unroll
        for (int batch2 = 0; batch2 < BATCH_PER_ITER_FORWARD; ++batch2) {
            if (batch2 < id_for_switch) {
                b_curr_acc[batch2] = curr_bias;
            }
        }

        for (int in_ch = 0; in_ch < in_c; ++in_ch) {

            for (int ker_x = 0; ker_x < max_x; ++ker_x) {
                const int curr_in_x = in_x + ker_x;

                const scalar_t weight = weights[in_ch][ker_x][out_ch][out_x];

                #pragma unroll
                for (int batch2 = 0; batch2 < BATCH_PER_ITER_FORWARD; ++batch2) {
                    if (batch2 < id_for_switch) {
                        b_input[batch2] = batch_in[batch + batch2][in_ch][curr_in_x];
                    }
                }

                #pragma unroll
                for (int batch2 = 0; batch2 < BATCH_PER_ITER_FORWARD; ++batch2) {
                    if (batch2 < id_for_switch) {
                        b_curr_acc[batch2] += weight * b_input[batch2];
                    }
                }
            }
        }

        #pragma unroll
        for (int batch2 = 0; batch2 < BATCH_PER_ITER_FORWARD; ++batch2) {
            if (batch2 < id_for_switch) {
                output[batch + batch2][out_ch][out_x] = b_curr_acc[batch2];
            }
        }
    }
}

template <typename scalar_t>
__global__ void backward(
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_input, // {N, in_c, in_s}
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad_weights, // {in_c, ker, out_c, out_s}
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> batch_in, // {N, in_c, in_s}
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> weights, // {in_c, ker, out_c, out_s}
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_output, // {N, out_c, out_s}
    const int batch_size,
    const int in_c,
    const int in_s,
    const int ker,
    const int out_c,
    const int out_s)
{
    const long in_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (in_idx >= in_c * in_s) {
        return;
    }

    int in_ch, in_x;
    get_item(in_idx, in_s, &in_ch, &in_x);

    int start_out_x;
    int end_out_x;

    if (in_s == ker) {
        start_out_x = 0;
        end_out_x = out_s;
    } else {
        start_out_x = round_div_int<long>(static_cast<long>(in_x - ker - 1) * (out_s - 1), in_s - ker);
        end_out_x = round_div_int<long>(static_cast<long>(in_x + ker + 1) * (out_s - 1), in_s - ker);
        start_out_x = std::max<int>(0, start_out_x);
        end_out_x = std::min<int>(out_s, end_out_x);
    }

    scalar_t b_input[BATCH_PER_ITER_BACKWARD];
    scalar_t b_grad_input[BATCH_PER_ITER_BACKWARD];
    scalar_t b_grad_output[BATCH_PER_ITER_BACKWARD];

    for (int batch = 0; batch < batch_size; batch += BATCH_PER_ITER_BACKWARD) {

        int id_for_switch = std::min(batch_size - batch, BATCH_PER_ITER_BACKWARD);

        #pragma unroll
        for (int batch2 = 0; batch2 < BATCH_PER_ITER_BACKWARD; ++batch2) {
            if (batch2 < id_for_switch) {
                b_input[batch2] = batch_in[batch + batch2][in_ch][in_x];
                b_grad_input[batch2] = 0;
            }
        }

        for (int out_x = start_out_x; out_x < end_out_x; ++out_x) {
            // Extract which ker_x this output pixel refers to (TODO verify sign)
            const int ker_x = in_x - round_div_int<long>(static_cast<long>(out_x) * (in_s - ker), out_s - 1);
            // Pass if the pixel does not match the one for this cuda kernel
            if (ker_x < 0 || ker_x >= ker) {
                continue;
            }

            for (int out_ch = 0; out_ch < out_c; ++out_ch) {

                scalar_t* grad_weight_ptr = &grad_weights[in_ch][ker_x][out_ch][out_x];

                // Access required data
                const scalar_t weight = weights[in_ch][ker_x][out_ch][out_x];
                const scalar_t grad_weight = *grad_weight_ptr;

                #pragma unroll
                for (int batch2 = 0; batch2 < BATCH_PER_ITER_BACKWARD; ++batch2) {
                    if (batch2 < id_for_switch) {
                        b_grad_output[batch2] = grad_output[batch + batch2][out_ch][out_x];
                    }
                }

                // Grad weight and Grad input add
                scalar_t grad_weight_add{grad_weight};

                #pragma unroll
                for (int batch2 = 0; batch2 < BATCH_PER_ITER_BACKWARD; ++batch2) {
                    if (batch2 < id_for_switch) {
                        scalar_t b2_grad_output = b_grad_output[batch2];
                        b_grad_input[batch2] += weight * b2_grad_output;
                        grad_weight_add += b_input[batch2] * b2_grad_output;
                    }
                }

                *grad_weight_ptr = grad_weight_add;
            }
        }

        // Grad input
        #pragma unroll
        for (int batch2 = 0; batch2 < BATCH_PER_ITER_BACKWARD; ++batch2) {
            if (batch2 < id_for_switch) {
                grad_input[batch + batch2][in_ch][in_x] = b_grad_input[batch2];
            }
        }
    }
}

} // namespace kernels::adjacent_1d

torch::Tensor _cuda_adjacent_1d_forward(
    torch::Tensor batch_in, // tensor of shape (N, in_c, in_s)
    int out_c,
    int out_s,
    int ker,
    torch::Tensor weights, // tensor of shape (in_c, ker, out_c, out_s)
    torch::Tensor bias) // tensor of shape (out_c, out_s)
{
    // Get values
    const int batch_size = batch_in.size(0);
    const int in_c = batch_in.size(1);
    const int in_s = batch_in.size(2);
    const long out_items = out_c * out_s;
    // Define threads & blocks
    const int threads = MAX_THREADS;
    const dim3 blocks(out_items / threads + (out_items % threads != 0));
    // Prepare data
    auto output = torch::empty(at::IntArrayRef({batch_size, out_c, out_s}), batch_in.options());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(batch_in.scalar_type(), "_cuda_kernel_adjacent_1d_forward", ([&] {
        using namespace kernels::adjacent_1d;

        forward<scalar_t><<<blocks, threads>>>(
            batch_in.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            weights.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            bias.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            output.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            batch_size,
            in_c,
            in_s,
            ker,
            out_c,
            out_s);
    }));

    // Check for errors
    auto error = cudaGetLastError();
    if (error) {
        throw std::runtime_error{"CUDA kernel error: " + std::string{cudaGetErrorString(error)}};
    }

    return output;
}

std::vector<torch::Tensor> _cuda_adjacent_1d_backward(
    torch::Tensor batch_in, // tensor of shape (N, in_c, in_s)
    int out_c,
    int out_s,
    int ker,
    torch::Tensor weights, // tensor of shape (in_c, ker, out_c, out_s)
    torch::Tensor bias, // tensor of shape (out_c, out_s)
    torch::Tensor grad_output) // tensor of shape (N, out_c, out_s)
{
    // Get values
    const int batch_size = batch_in.size(0);
    const int in_c = batch_in.size(1);
    const int in_s = batch_in.size(2);
    const long in_items = in_c * in_s;
    // Define threads & blocks
    const int threads = MAX_THREADS;
    const dim3 blocks(in_items / threads + (in_items % threads != 0));
    // Prepare data
    auto grad_input = torch::zeros_like(batch_in, batch_in.options());
    auto grad_weights = torch::zeros_like(weights, batch_in.options());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(batch_in.scalar_type(), "_cuda_kernel_adjacent_1d_backward", ([&] {
        using namespace kernels::adjacent_1d;

        backward<scalar_t><<<blocks, threads>>>(
            grad_input.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            grad_weights.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            batch_in.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            weights.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            grad_output.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            batch_size,
            in_c,
            in_s,
            ker,
            out_c,
            out_s);
    }));

    // Check for errors
    auto error = cudaGetLastError();
    if (error) {
        throw std::runtime_error{"CUDA kernel error: " + std::string{cudaGetErrorString(error)}};
    }

    // Calculate bias which is the sum of the output grads along batches
    auto grad_bias = torch::sum(grad_output, 0);

    return {grad_input, grad_weights, grad_bias};
}


#include <torch/extension.h>

#include <thrust/tuple.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <THC/THCAtomics.cuh>

#include <ATen/AccumulateType.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <c10/cuda/CUDAMathCompat.h>

#include <vector>


torch::Tensor _cuda_pish_forward(torch::Tensor batch_in, torch::Tensor weight)
{
    // Prepare data
    auto output = torch::zeros_like(batch_in, batch_in.options());

    float weight0 = weight[0].item<float>();
    float weight1 = weight[1].item<float>();

    using namespace at;
    using namespace at::native;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        batch_in.scalar_type(),
        "_cuda_pish_forward_kernel",
        [&] {
            using namespace at;
            using namespace at::native;
            auto iter = TensorIteratorConfig().add_output(output).add_input(batch_in).build();

            gpu_kernel(iter, [weight0, weight1]GPU_LAMBDA(scalar_t x) -> scalar_t {
                namespace math = c10::cuda::compat;
                // Param neg
                scalar_t param_neg = weight0 * x;
                scalar_t bound_neg = math::max(x, param_neg);
                // Mish calculation
                using T_ACC = acc_type<scalar_t, true>;
                const T_ACC x_acc = static_cast<T_ACC>(x);
                scalar_t mish = x_acc * math::tanh(math::log1p(math::exp(x_acc)));
                // Fuse
                return math::min(weight1 * mish, bound_neg);
            });

        }
    );

    // Check for errors
    auto error = cudaGetLastError();
    if (error) {
        throw std::runtime_error{"CUDA kernel error: " + std::string{cudaGetErrorString(error)}};
    }

    return output;
}

std::vector<torch::Tensor> _cuda_pish_backward(torch::Tensor batch_in,
                                               torch::Tensor weight,
                                               torch::Tensor grad_output)
{
    // Prepare data
    auto grad_input = torch::zeros_like(batch_in, batch_in.options());
    auto grad_weight_collector1 = torch::zeros_like(batch_in, batch_in.options());
    auto grad_weight_collector2 = torch::zeros_like(batch_in, batch_in.options());

    float weight0 = weight[0].item<float>();
    float weight1 = weight[1].item<float>();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        batch_in.scalar_type(),
        "_cuda_pish_backward_kernel",
        [&] {
            using namespace at;
            using namespace at::native;
            auto iter = TensorIteratorConfig()
                .add_output(grad_input)
                .add_output(grad_weight_collector1)
                .add_output(grad_weight_collector2)
                .add_input(grad_output)
                .add_input(batch_in)
                .build();

            gpu_kernel_multiple_outputs(iter, [weight0, weight1]GPU_LAMBDA(scalar_t dy, scalar_t x) -> thrust::tuple<scalar_t, scalar_t, scalar_t> {
                namespace math = c10::cuda::compat;
                // Negative bound
                scalar_t param_neg = weight0 * x;
                scalar_t bound_neg = math::max(x, param_neg);
                // Mish calculation
                using T_ACC = acc_type<scalar_t, true>;
                const T_ACC x_acc = static_cast<T_ACC>(x);
                scalar_t mish = x_acc * math::tanh(math::log1p(math::exp(x_acc)));
                scalar_t m_mish = weight1 * mish;

                // Derivatives
                if (bound_neg <= m_mish) {
                    // Derivate negative bound
                    if (param_neg >= x) {
                        return {weight0 * dy, x * dy, 0};
                    }
                    return {dy, 0, 0};
                }

                // Derivate mish
                const T_ACC dy_acc = static_cast<T_ACC>(dy);
                const T_ACC s_acc = T_ACC(1) / (T_ACC(1) + math::exp(-x_acc));
                const T_ACC t_acc = math::tanh(math::log1p(math::exp(x_acc)));
                scalar_t d_mish = dy_acc * (t_acc + x_acc * s_acc * (T_ACC(1) - t_acc * t_acc));

                return {weight1 * d_mish, 0, mish * dy};
            });

        }
    );

    auto grad_weight = torch::zeros_like(weight, weight.options());
    grad_weight[0].fill_(grad_weight_collector1.sum());
    grad_weight[1].fill_(grad_weight_collector2.sum());

    // Check for errors
    auto error = cudaGetLastError();
    if (error) {
        throw std::runtime_error{"CUDA kernel error: " + std::string{cudaGetErrorString(error)}};
    }

    return {grad_input, grad_weight};
}


torch::Tensor _cuda_wish_forward(torch::Tensor batch_in, torch::Tensor weight, float gate)
{
    // Prepare data
    auto output = torch::zeros_like(batch_in, batch_in.options());

    float weight0 = weight[0].item<float>();
    float weight1 = weight[1].item<float>();

    using namespace at;
    using namespace at::native;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        batch_in.scalar_type(),
        "_cuda_wish_forward_kernel",
        [&] {
            using namespace at;
            using namespace at::native;
            auto iter = TensorIteratorConfig().add_output(output).add_input(batch_in).build();

            gpu_kernel(iter, [gate, weight0, weight1]GPU_LAMBDA(scalar_t x) -> scalar_t {
                namespace math = c10::cuda::compat;
                using T_ACC = acc_type<scalar_t, true>;
                // Mish calculation
                const T_ACC x_acc = static_cast<T_ACC>(x);
                const T_ACC mish = x_acc * math::tanh(math::log1p(math::exp(x_acc)));
                // Fuse
                if (x < gate) {
                    return weight0 * (x - gate) + weight1 * mish;
                } else {
                    return weight1 * mish;
                }
            });
        }
    );

    // Check for errors
    auto error = cudaGetLastError();
    if (error) {
        throw std::runtime_error{"CUDA kernel error: " + std::string{cudaGetErrorString(error)}};
    }

    return output;
}

std::vector<torch::Tensor> _cuda_wish_backward(torch::Tensor batch_in,
                                               torch::Tensor weight,
                                               torch::Tensor grad_output,
                                               float gate)
{
    // Prepare data
    auto grad_input = torch::zeros_like(batch_in, batch_in.options());
    auto grad_weight_collector0 = torch::zeros_like(batch_in, batch_in.options());
    auto grad_weight_collector1 = torch::zeros_like(batch_in, batch_in.options());

    float weight0 = weight[0].item<float>();
    float weight1 = weight[1].item<float>();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        batch_in.scalar_type(),
        "_cuda_wish_backward_kernel",
        [&] {
            using namespace at;
            using namespace at::native;
            auto iter = TensorIteratorConfig()
                .add_output(grad_input)
                .add_output(grad_weight_collector0)
                .add_output(grad_weight_collector1)
                .add_input(grad_output)
                .add_input(batch_in)
                .build();

            gpu_kernel_multiple_outputs(iter, [gate, weight0, weight1]GPU_LAMBDA(scalar_t dy, scalar_t x) -> thrust::tuple<scalar_t, scalar_t, scalar_t> {
                namespace math = c10::cuda::compat;
                // Mish calculation
                using T_ACC = acc_type<scalar_t, true>;
                const T_ACC x_acc = static_cast<T_ACC>(x);
                const T_ACC t_acc = math::tanh(math::log1p(math::exp(x_acc)));
                const T_ACC mish = x_acc * t_acc;
                const T_ACC m_mish = weight1 * mish;
                // Derivate mish
                const T_ACC dy_acc = static_cast<T_ACC>(dy);
                const T_ACC s_acc = T_ACC(1) / (T_ACC(1) + math::exp(-x_acc));
                const T_ACC d_mish = dy_acc * (t_acc + x_acc * s_acc * (T_ACC(1) - t_acc * t_acc));

                if (x < gate) {
                    return {weight1 * d_mish, (x - gate) * dy, mish * dy};
                } else {
                    return {weight1 * d_mish, 0, mish * dy};
                }
            });
        }
    );

    auto grad_weight = torch::zeros_like(weight, weight.options());
    grad_weight[0].fill_(grad_weight_collector0.sum());
    grad_weight[1].fill_(grad_weight_collector1.sum());

    // Check for errors
    auto error = cudaGetLastError();
    if (error) {
        throw std::runtime_error{"CUDA kernel error: " + std::string{cudaGetErrorString(error)}};
    }

    return {grad_input, grad_weight};
}

#include <torch/extension.h>

#include <iostream>
#include <vector>

// CUDA checkers

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// CUDA declarations

torch::Tensor _cuda_pish_forward(torch::Tensor batch_in, torch::Tensor weight);

std::vector<torch::Tensor> _cuda_pish_backward(torch::Tensor batch_in,
                                               torch::Tensor weight,
                                               torch::Tensor grad_output);

torch::Tensor _cuda_wish_forward(torch::Tensor batch_in, torch::Tensor weight, float gate);

std::vector<torch::Tensor> _cuda_wish_backward(torch::Tensor batch_in,
                                               torch::Tensor weight,
                                               torch::Tensor grad_output,
                                               float gate);

namespace wish::cuda {

torch::Tensor forward(torch::Tensor batch_in, torch::Tensor weight, float gate)
{
    CHECK_INPUT(batch_in);
    CHECK_INPUT(weight);

    return _cuda_wish_forward(batch_in, weight, gate);
}

std::vector<torch::Tensor> backward(torch::Tensor batch_in,
                                    torch::Tensor weight,
                                    torch::Tensor grad_output,
                                    float gate)
{
    CHECK_INPUT(batch_in);
    CHECK_INPUT(weight);
    CHECK_INPUT(grad_output);

    return _cuda_wish_backward(batch_in, weight, grad_output, gate);
}

} // namespace pish::cuda

namespace pish::cuda {

torch::Tensor forward(torch::Tensor batch_in, torch::Tensor weight)
{
    CHECK_INPUT(batch_in);
    CHECK_INPUT(weight);

    return _cuda_pish_forward(batch_in, weight);
}

std::vector<torch::Tensor> backward(torch::Tensor batch_in,
                                    torch::Tensor weight,
                                    torch::Tensor grad_output)
{
    CHECK_INPUT(batch_in);
    CHECK_INPUT(weight);
    CHECK_INPUT(grad_output);

    return _cuda_pish_backward(batch_in, weight, grad_output);
}

} // namespace pish::cuda

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cuda_pish_forward", &pish::cuda::forward, "Pish forward (cuda)");
    m.def("cuda_pish_backward", &pish::cuda::backward, "Pish backward (cuda)");
    m.def("cuda_wish_forward", &wish::cuda::forward, "Wish forward (cuda)");
    m.def("cuda_wish_backward", &wish::cuda::backward, "Wish backward (cuda)");
}

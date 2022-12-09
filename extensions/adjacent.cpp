#include <torch/extension.h>

#include <iostream>
#include <vector>

// CUDA checkers

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// CUDA declarations

torch::Tensor _cuda_adjacent_2d_forward(torch::Tensor batch_in,
                                                 int out_c,
                                                 int out_h,
                                                 int out_w,
                                                 int ker_h,
                                                 int ker_w,
                                                 torch::Tensor weights,
                                                 torch::Tensor bias);

std::vector<torch::Tensor> _cuda_adjacent_2d_backward(torch::Tensor batch_in,
                                                               int out_c,
                                                               int out_h,
                                                               int out_w,
                                                               int ker_h,
                                                               int ker_w,
                                                               torch::Tensor weights,
                                                               torch::Tensor bias,
                                                               torch::Tensor grad_output);

torch::Tensor _cuda_adjacent_1d_forward(torch::Tensor batch_in,
                                        int out_c,
                                        int out_s,
                                        int ker,
                                        torch::Tensor weights,
                                        torch::Tensor bias);

std::vector<torch::Tensor> _cuda_adjacent_1d_backward(torch::Tensor batch_in,
                                                      int out_c,
                                                      int out_s,
                                                      int ker_s,
                                                      torch::Tensor weights,
                                                      torch::Tensor bias,
                                                      torch::Tensor grad_output);

// C++ functions

template <typename T>
inline T round_div_int(T val, T div)
{
    T output = (val + div / 2) / div;
    return output;
}

namespace adjacent_2d {

namespace cpu {

torch::Tensor forward(torch::Tensor batch_in,
                      int out_c,
                      int out_h,
                      int out_w,
                      int ker_h,
                      int ker_w,
                      torch::Tensor weights,
                      torch::Tensor bias)
{
    using namespace torch::indexing;

    // Shortcuts
    int batch_size = batch_in.sizes()[0];
    int in_c = batch_in.sizes()[1];
    int in_h = batch_in.sizes()[2];
    int in_w = batch_in.sizes()[3];

    // Create an input data to be used in the final operation
    int flat_ker_in_dim = in_c * ker_h * ker_w;
    auto tmp_in = torch::empty(
        at::IntArrayRef({out_h, out_w, batch_size, in_c, ker_h, ker_w}),
        batch_in.options()
    );

    // Iterate over output pixels
    for(int y = 0; y < out_h; ++y) {
        int in_start_y = round_div_int<int>(static_cast<long>(y) * (in_h - ker_h), out_h - 1);
        int in_end_y = in_start_y + ker_h;

        for(int x = 0; x < out_w; ++x) {
            int in_start_x = round_div_int<int>(static_cast<long>(x) * (in_w - ker_w), out_w - 1);
            int in_end_x = in_start_x + ker_w;

            auto local_in_index = batch_in.index(
                {Slice(), Slice(), Slice(in_start_y, in_end_y), Slice(in_start_x, in_end_x)});

            tmp_in.index_put_({y, x}, local_in_index);
        }
    }

    // Calculate the input * weight
    tmp_in = tmp_in.view({out_h * out_w, batch_size, flat_ker_in_dim});
    auto weights_reorganized = weights.view({flat_ker_in_dim, out_c, out_h * out_w});
    weights_reorganized = weights_reorganized.permute({2, 0, 1});

    auto res = torch::empty({out_h * out_w, batch_size, out_c}, batch_in.options());
    torch::bmm_out(res, tmp_in, weights_reorganized);

    // Init output with bias
    auto output = torch::empty(
        at::IntArrayRef({batch_size, out_c, out_h, out_w}),
        batch_in.options()
    );
    for (int i = 0; i < batch_size; ++i) {
        output.index_put_({i}, bias);
    }

    // Add the result to the output
    res = res.permute({1, 2, 0});
    output.add_(res.view({batch_size, out_c, out_h, out_w}));

    return output;
}

std::vector<torch::Tensor> backward(torch::Tensor batch_in,
                                    int out_c,
                                    int out_h,
                                    int out_w,
                                    int ker_h,
                                    int ker_w,
                                    torch::Tensor weights,
                                    torch::Tensor bias,
                                    torch::Tensor grad_output)
{
    using namespace torch::indexing;

    // Shortcuts
    int batch_size = batch_in.sizes()[0];
    int in_c = batch_in.sizes()[1];
    int in_h = batch_in.sizes()[2];
    int in_w = batch_in.sizes()[3];

    // Create output tensors that will be filled later
    auto grad_input = torch::zeros_like(batch_in, batch_in.options());
    auto grad_weights = torch::zeros_like(weights, batch_in.options());
    auto grad_bias = torch::zeros_like(bias, batch_in.options());

    // For retrocomp, flat the grad weight
    int flat_ker_in_dim = in_c * ker_h * ker_w;
    grad_weights = grad_weights.view({flat_ker_in_dim, out_c, out_h, out_w});
    weights = weights.view({flat_ker_in_dim, out_c, out_h, out_w});

    // Iterate over output pixels
    for(int y = 0; y < out_h; ++y) {
        int in_start_y = round_div_int<int>(static_cast<long>(y) * (in_h - ker_h), out_h - 1);
        int in_end_y = in_start_y + ker_h;
        for(int x = 0; x < out_w; ++x) {
            // Extract input kernel and weights
            int in_start_x = round_div_int<int>(static_cast<long>(x) * (in_w - ker_w), out_w - 1);
            int in_end_x = in_start_x + ker_w;

            auto local_in_index = batch_in.index(
                {Slice(), Slice(), Slice(in_start_y, in_end_y), Slice(in_start_x, in_end_x)});
            auto local_weights = weights.index({Slice(), Slice(), y, x}).t();

            // Right now there is an issue with view() for splitted tensors, this
            // fix the issue but also instanciate undesired memory.
            auto local_in = local_in_index.clone();

            // Set weights such that it matches the local_in tensor.
            int flat_dimension = ker_w * ker_h * in_c;
            local_in = local_in.view({batch_size, flat_dimension});

            // Calculate the gradients
            auto local_grad_output = grad_output.index({Slice(), Slice(), y, x});

            // Inputs gradients
            auto local_grad_input = local_grad_output.mm(local_weights);
            local_grad_input = local_grad_input.view({batch_size, in_c, ker_h, ker_w});
            grad_input.index(
                {Slice(), Slice(), Slice(in_start_y, in_end_y), Slice(in_start_x, in_end_x)}
            ).add_(local_grad_input);
            // Weights gradients
            auto local_grad_weights = local_grad_output.t().mm(local_in);
            grad_weights.index_put_({Slice(), Slice(), y, x}, local_grad_weights.t());
            // Bias gradients
            grad_bias.index_put_({Slice(), y, x}, local_grad_output.sum(0));
        }
    }

    // For retrocomp, unflat the grad weight
    grad_weights = grad_weights.view({in_c, ker_h, ker_w, out_c, out_h, out_w});

    return {
        grad_input,
        grad_weights,
        grad_bias
    };
}

} // namespace cpu

namespace cuda {

torch::Tensor forward(torch::Tensor batch_in,
                      int out_c,
                      int out_h,
                      int out_w,
                      int ker_h,
                      int ker_w,
                      torch::Tensor weights,
                      torch::Tensor bias)
{
    CHECK_INPUT(batch_in);
    CHECK_INPUT(weights);
    CHECK_INPUT(bias);

    return _cuda_adjacent_2d_forward(
        batch_in, out_c, out_h, out_w, ker_h, ker_w, weights, bias
    );
}

std::vector<torch::Tensor> backward(torch::Tensor batch_in,
                                    int out_c,
                                    int out_h,
                                    int out_w,
                                    int ker_h,
                                    int ker_w,
                                    torch::Tensor weights,
                                    torch::Tensor bias,
                                    torch::Tensor grad_output)
{
    CHECK_INPUT(batch_in);
    CHECK_INPUT(weights);
    CHECK_INPUT(bias);
    CHECK_INPUT(grad_output);

    return _cuda_adjacent_2d_backward(
        batch_in, out_c, out_h, out_w, ker_h, ker_w, weights, bias, grad_output
    );
}

} // namespace cuda

} // namespace adjacent_2d


namespace adjacent_1d::cuda {

torch::Tensor forward(torch::Tensor batch_in,
                      int out_c,
                      int out_s,
                      int ker,
                      torch::Tensor weights,
                      torch::Tensor bias)
{
    CHECK_INPUT(batch_in);
    CHECK_INPUT(weights);
    CHECK_INPUT(bias);

    return _cuda_adjacent_1d_forward(batch_in, out_c, out_s, ker, weights, bias);
}

std::vector<torch::Tensor> backward(torch::Tensor batch_in,
                                    int out_c,
                                    int out_s,
                                    int ker,
                                    torch::Tensor weights,
                                    torch::Tensor bias,
                                    torch::Tensor grad_output)
{
    CHECK_INPUT(batch_in);
    CHECK_INPUT(weights);
    CHECK_INPUT(bias);
    CHECK_INPUT(grad_output);

    return _cuda_adjacent_1d_backward(batch_in, out_c, out_s, ker, weights, bias, grad_output);
}

} // namespace adjacent_1d::cuda

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("adjacent_2d_forward", &adjacent_2d::cpu::forward, "Adjacent2d forward");
    m.def("adjacent_2d_backward", &adjacent_2d::cpu::backward, "Adjacent2d backward");
    m.def("cuda_adjacent_2d_forward", &adjacent_2d::cuda::forward, "Adjacent2d forward (cuda)");
    m.def("cuda_adjacent_2d_backward", &adjacent_2d::cuda::backward, "Adjacent2d backward (cuda)");
    m.def("cuda_adjacent_1d_forward", &adjacent_1d::cuda::forward, "Adjacent1d forward (cuda)");
    m.def("cuda_adjacent_1d_backward", &adjacent_1d::cuda::backward, "Adjacent1d backward (cuda)");
}

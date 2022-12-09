#include <torch/extension.h>

#include <iostream>

///
/// Return the Gray code of n
///
uint64_t graycode(uint64_t n)
{
    return (n ^ (n >> 1));
}

///
/// Fill a tensor with 0 and 1 representing the provided binary value.
///
void feed_binary_tensor(uint64_t bin_value, torch::Tensor& t_out)
{
    size_t bitsize = t_out.sizes()[0];
    uint64_t mask = 1;
    for (size_t i = bitsize - 1; ; --i) {
        bool bitval = bin_value & mask;
        if (bitval) {
            t_out[i] = 1;
        } else {
            t_out[i] = 0;
        }
        if (i == 0) break;
        mask = mask << 1;
    }
}

///
/// Fill a batch of tensors with 0 and 1 representing the gray code of the values provided in
/// another tensor.
///
void layer_graycode(const torch::Tensor& t_val, torch::Tensor& t_out)
{
    size_t batch_size = t_out.sizes()[0];
    for (size_t b = 0; b < batch_size; ++b) {
        uint64_t bin_value = t_val[b][0].item<long>();
        uint64_t gray_code = graycode(bin_value);
        torch::Tensor t_out_spec = t_out[b];
        feed_binary_tensor(gray_code, t_out_spec);
    }
}

///
/// Fill a batch of tensors with 0 and 1 representing the binary of the values provided in
/// another tensor.
///
void layer_binary(const torch::Tensor& t_val, torch::Tensor& t_out)
{
    size_t batch_size = t_out.sizes()[0];
    for (size_t b = 0; b < batch_size; ++b) {
        uint64_t bin_value = t_val[b][0].item<long>();
        torch::Tensor t_out_spec = t_out[b];
        feed_binary_tensor(bin_value, t_out_spec);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("graycode", &graycode, "Return the Gray code of n");
    m.def("feed_binary_tensor", &feed_binary_tensor, "Fill a tensor with 0 and 1 representing the provided binary value.");
    m.def("layer_graycode", &layer_graycode);
    m.def("layer_binary", &layer_binary);
}

#include <torch/extension.h>

#include <iostream>
#include <vector>
#include <cerrno>
#include <cstring>

extern "C" {
#include <libvmaf/libvmaf.h>
}

void hello_world()
{
    std::cout << "Hello world" << std::endl;
}

///
/// Feed picture for 444p format and 8 bpc
///
static void feed_picture_with_tensor_yuv444n(const torch::Tensor& t, VmafPicture *dst)
{
    const unsigned int height = t.sizes()[1];
    const unsigned int width = t.sizes()[2];
    const unsigned int datasize = width * height * sizeof(uint8_t);

    for (unsigned int c = 0; c < 3; ++c) {
        uint8_t* dst_data = static_cast<uint8_t*>(dst->data[c]);
        uint8_t* src_data = t[c].data_ptr<uint8_t>();
        std::memcpy(dst_data, src_data, datasize);
    }
}

class VmafCompute {
public:
    VmafCompute(const std::string& model_path):
        vmaf_{nullptr}, model_{nullptr}, pix_fmt_{VMAF_PIX_FMT_YUV444P}, next_frame_id_{0}
    {
        int err;

        // Initialise context
        VmafConfiguration cfg = {
            .log_level = VMAF_LOG_LEVEL_WARNING,
            .n_threads = 0,
            .n_subsample = 0,
            .cpumask = 0,
        };

        err = vmaf_init(&vmaf_, cfg);

        if (err) {
            throw std::runtime_error{
                std::string{"problem initializing VMAF context: "} + std::strerror(-err)
            };
        }

        // Initialise model
        VmafModelConfig model_cfg = {
            .name = "vmaf",
            .flags = static_cast<enum VmafModelFlags>(VMAF_MODEL_FLAGS_DEFAULT),
        };

        // TODO: turn that into vmaf_model_load_from_path
        err = vmaf_model_load(&model_, &model_cfg, model_path.c_str());
        if (err) {
            vmaf_close(vmaf_);
            throw std::runtime_error{std::string{"problem loading model: "} + std::strerror(-err)};
        }
        err = vmaf_use_features_from_model(vmaf_, model_);
        if (err) {
            vmaf_model_destroy(model_);
            vmaf_close(vmaf_);

            throw std::runtime_error{
                std::string{"problem loading feature extractors from model file ("} + model_path +
                "): " + std::strerror(-err)
            };
        }
    }

    unsigned int get_frame_count() const
    {
        return next_frame_id_;
    }

    void add_frame(torch::Tensor ref, torch::Tensor dist)
    {
        unsigned int height = ref.sizes()[1];
        unsigned int width = ref.sizes()[2];
        unsigned int bpc = 8;

        int err;

        // Allocate pictures
        VmafPicture pic_ref, pic_dist;
        err = vmaf_picture_alloc(&pic_ref, pix_fmt_, bpc, width, height);
        err |= vmaf_picture_alloc(&pic_dist, pix_fmt_, bpc, width, height);

        if (err) {
            vmaf_picture_unref(&pic_ref);
            vmaf_picture_unref(&pic_dist);
            throw std::runtime_error{
                std::string{"problem allocating picture memory: "} + std::strerror(-err)
            };
        }

        // Feed data into the allocated images
        feed_picture_with_tensor_yuv444n(ref, &pic_ref);
        feed_picture_with_tensor_yuv444n(dist, &pic_dist);

        // Feed images in context
        err = vmaf_read_pictures(vmaf_, &pic_ref, &pic_dist, next_frame_id_++);
        if (err) {
            throw std::runtime_error{
                std::string{"problem reading pictures: "} + std::strerror(-err)
            };
        }
    }

    void flush()
    {
        int err = vmaf_read_pictures(vmaf_, NULL, NULL, 0);
        if (err) {
            throw std::runtime_error{
                std::string{"problem flushing context: "} + std::strerror(-err)
            };
        }
    }

    double score_at_index(unsigned int index)
    {
        double vmaf_score;
        int err = vmaf_score_at_index(vmaf_, model_, &vmaf_score, index);
        if (err) {
            throw std::runtime_error{
                std::string{"problem generating VMAF score: "} + std::strerror(-err)
            };
        }
        return vmaf_score;
    }

    double pool_mean(unsigned int index)
    {
        double blabla;
        int err = vmaf_score_pooled(vmaf_, model_, VMAF_POOL_METHOD_MEAN, &blabla, 0, index);
        if (err) {
            throw std::runtime_error{
                std::string{"problem generating pooled VMAF score: "} + std::strerror(-err)
            };
        }
        return blabla;
    }

    void feed_scores(torch::Tensor& scores)
    {
        for (unsigned int frame = 0; frame < next_frame_id_; ++frame) {
            scores[frame] = score_at_index(frame);
        }
    }

    ~VmafCompute()
    {
        vmaf_model_destroy(model_);
        vmaf_close(vmaf_);
    }

private:
    VmafContext* vmaf_;
    VmafModel* model_;
    enum VmafPixelFormat pix_fmt_;
    unsigned int next_frame_id_;
};


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hello_world", &hello_world, "Hello world function");

    py::class_<VmafCompute>(m, "VmafCompute")
        .def(py::init<const std::string &>())
        .def("get_frame_count", &VmafCompute::get_frame_count)
        .def("add_frame", &VmafCompute::add_frame)
        .def("flush", &VmafCompute::flush)
        .def("score_at_index", &VmafCompute::score_at_index)
        .def("pool_mean", &VmafCompute::pool_mean)
        .def("feed_scores", &VmafCompute::feed_scores)
    ;
}

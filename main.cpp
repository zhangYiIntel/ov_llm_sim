#include <iostream>
#include <algorithm>
#include <ie_core.hpp>
#include <openvino/opsets/opset8.hpp>
#include <openvino/pass/serialize.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>
#include "openvino/openvino.hpp"
#include <thread>

using namespace ov;

int main(int args, char *argv[]) {
    if (args < 2) {
        std::cout << "Please provide model path" << std::endl;
        exit(-1);
    }
    std::string model_path(argv[1]);
    // std::string model_path("C:/\Users/\zhangyi7/\models_analysis/\stateful/\stateful_simple_chatglm3.xml");
    ov::Core core;
    auto ov_version = ov::get_openvino_version();
    std::cout << "OPENVINO|VERSION|" << ov_version << std::endl;
    std::cout << "Gong to load model|" << model_path << std::endl;
    std::shared_ptr<ov::Model> model;
    try {
        model = core.read_model(model_path);
    } catch(std::exception& ex) {
        std::cout << ex.what() << std::endl;
    }
    using namespace InferenceEngine;
    auto enc_inputs = model->inputs();
    auto& features = enc_inputs[0];
    std::cout << "Finish model loading" << std::endl;
    // core.set_property("CPU", ov::num_streams(1));
    // core.set_property("CPU", ov::inference_num_threads(1));
    // core.set_property("CPU", ov::affinity(ov::Affinity::NONE));
    // core.set_property("GPU", ov::hint::inference_precision(getenv("ENABLE_BF16") ? ov::element::bf16 : ov::element::f32));
   
    auto compiled_model = core.compile_model(model, "CPU", {
        {"PERF_COUNT", "YES"}
    });

    auto type = ov::element::i64;
    auto llm_infer_ = compiled_model.create_infer_request();
    std::vector<int64_t> input_ids = {2,2,2,2,2};

    auto encoder_inputs_ = model->inputs();

    //input_cache
    ov::Shape input_ids_shape = {1, 5};
    ov::Tensor input_ids_ov = ov::Tensor(type, input_ids_shape, input_ids.data());

    // attention_mask
    std::vector<int64_t> attn_mask(10, 1);
    ov::Shape attn_mask_shape = {1, 5};
    ov::Tensor attn_mask_ov = ov::Tensor(type, attn_mask_shape, attn_mask.data());


    std::vector<int64_t> position_ids{0,1,2,3,4,5,6,7,8,9,10};
    ov::Shape position_ids_shape = {1, 5};
    ov::Tensor position_ids_ov = ov::Tensor(type, position_ids_shape, position_ids.data());

    // fake hidden stat

    std::cout << "Going to Iter" << std::endl;

    for(size_t i = 0; i < 2; i++) {
        std::cout << "The " << i+1 << " token" << std::endl;
        if (i != 0) {
            attn_mask_ov = ov::Tensor(type, {1, 5 + i}, attn_mask.data());
            position_ids_ov = ov::Tensor(type, {1, 5 + i}, position_ids.data());
            std::vector<int> tmp{1};
            input_ids_ov = ov::Tensor(type, {1, 1}, tmp.data());
        }
        int idx = 0;
        for (auto& input : encoder_inputs_) {
            std::string name = input.get_node()->get_friendly_name();
            std::cout << "set input " << name << std::endl;
            if (name == "input_ids") {
                llm_infer_.set_input_tensor(idx++, input_ids_ov);
            } else if (name == "attention_mask") {
                llm_infer_.set_input_tensor(idx++, attn_mask_ov);
            } else if (name == "position_ids") {
                llm_infer_.set_input_tensor(idx++, position_ids_ov);
            }
        }

        try {
            llm_infer_.infer();
        } catch(std::exception& ex) {
            std::cout << "infer" << ex.what() << std::endl;
        }
        std::cout << "Infer Finished" << std::endl;
        auto out = llm_infer_.get_output_tensor(0);
        std::cout << "output_shape|" << out.get_shape() << std::endl;
        float* outPtr = out.data<float>();
        for (size_t i = 0; i < 5; i++) {
            for(size_t j = 0; j < 5; j++) {
                std::cout << outPtr[i*4096 + j] << ",";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    static int count = 0;
    ov::serialize(compiled_model.get_runtime_model(),
        "llm_exec_graph_"+std::to_string(count++)+".xml");
}
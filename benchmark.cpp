#include <iostream>
#include <algorithm>
#include <openvino/opsets/opset8.hpp>
#include <openvino/pass/serialize.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>
#include "openvino/openvino.hpp"
#include <thread>
#include <numeric>
#include <chrono>

using namespace ov;

int main(int args, char *argv[]) {
    if (args < 2) {
        std::cout << "please provide mode path" << std::endl;
        exit(-1);
    }
    std::string model_path(argv[1]);
    ov::Core core;
    std::shared_ptr<ov::Model> model = core.read_model(model_path);
    auto enc_inputs = model->inputs();
    auto& features = enc_inputs[0];
    auto ov_version = ov::get_openvino_version();
    size_t seq_len = args >= 3 ? std::stoi(std::string(argv[2])) : 2016;
    std::cout << "OPENVINO|VERSION|" << ov_version << std::endl;
    core.set_property("CPU", ov::num_streams(1));
    // core.set_property("CPU", ov::inference_num_threads(1));
    // core.set_property("CPU", ov::affinity(ov::Affinity::NONE));
    core.set_property("CPU", ov::hint::inference_precision(getenv("DISABLE_BF16") ? ov::element::f32 : ov::element::bf16));
    bool enable_bf16 = true;
    if (enable_bf16) {
        ov::preprocess::PrePostProcessor preproc(model);
        ov::preprocess::OutputInfo& output = preproc.output(0);
        output.tensor().set_element_type(ov::element::bf16);
        model = preproc.build();
    }
    auto compiled_model = core.compile_model(model, "CPU", {
        {"PERF_COUNT", "YES"},
        // {"DYNAMIC_QUANTIZATION_GROUP_SIZE", "256"}
    });
    auto llm_infer_ = compiled_model.create_infer_request();
    auto type = ov::element::i64;
    std::cout << "Going to Iter" << std::endl;
    for (size_t count = 0; count < 2; count++) {
        size_t batch_size = 8;
        std::vector<int64_t> input_ids(batch_size * (seq_len + 100), 2);

        auto encoder_inputs_ = model->inputs();
        // fake hidden state
        //input_cache

        ov::Shape input_ids_shape = {batch_size, seq_len};
        ov::Tensor input_ids_ov = ov::Tensor(type, input_ids_shape, input_ids.data());

        // attention_mask
        std::vector<int64_t> attn_mask(batch_size*(seq_len + 100), 1);
        ov::Shape attn_mask_shape = {batch_size, seq_len};
        ov::Tensor attn_mask_ov = ov::Tensor(type, attn_mask_shape, attn_mask.data());


        std::vector<int64_t> position_ids(batch_size*(seq_len + 100));
        for (size_t i = 0; i < batch_size; i++) {
            std::iota(position_ids.begin() + i * (seq_len + 100), position_ids.begin() + (i + 1) * (seq_len + 100), 0);
        }
        
        ov::Shape position_ids_shape = {batch_size, seq_len};
        ov::Tensor position_ids_ov = ov::Tensor(type, position_ids_shape, position_ids.data());
        
        std::vector<int32_t> beam_idx(batch_size, 0);
        std::iota(beam_idx.begin(), beam_idx.end(), 0);
        ov::Shape beam_idx_shape = {batch_size};
        ov::Tensor beam_idx_ov = ov::Tensor(ov::element::i32, beam_idx_shape, beam_idx.data());
        size_t idx = 0;
        for (auto& input : encoder_inputs_) {
            std::string name = input.get_node()->get_friendly_name();
            std::cout << "set input " << name << std::endl;
            if (name == "input_ids") {
                llm_infer_.set_input_tensor(idx++, input_ids_ov);
            } else if (name == "attention_mask") {
                llm_infer_.set_input_tensor(idx++, attn_mask_ov);
            } else if (name == "position_ids") {
                llm_infer_.set_input_tensor(idx++, position_ids_ov);
            } else if (name == "beam_idx") {
                llm_infer_.set_input_tensor(idx++, beam_idx_ov);
            }
        }
        for (size_t i = 0; i < 10; i++) {
            try {
                llm_infer_.infer();
            } catch(std::exception& ex) {
                std::cout << "infer" << ex.what() << std::endl;
            }
            if (count > 0)
                std::this_thread::sleep_for(std::chrono::milliseconds(900));
        }
    }

    static int count = 0;
    ov::serialize(compiled_model.get_runtime_model(),
        "llm_exec_graph_"+std::to_string(count++)+".xml");
}


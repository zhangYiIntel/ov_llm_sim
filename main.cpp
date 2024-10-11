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
    size_t seq_len = args >= 3 ? std::stoi(std::string(argv[2])) : 2064;
    bool is_beam_search = args >= 4 ? true : false;
    std::cout << "OPENVINO|VERSION|" << ov_version << std::endl;
    std::string search_type = is_beam_search ? "Beam Search" : "Greedy Search";
    std::cout << "Search Type|" << search_type << std::endl;
    core.set_property("CPU", ov::num_streams(1));
    // core.set_property("CPU", ov::inference_num_threads(1));
    // core.set_property("CPU", ov::affinity(ov::Affinity::NONE));
    core.set_property("CPU", ov::hint::inference_precision(getenv("DISABLE_BF16") ? ov::element::f32 : ov::element::bf16));
    //core.set_property("CPU", ov::hint::inference_precision(ov::element::bf16));
    auto compiled_model = core.compile_model(model, "CPU", {
        {"PERF_COUNT", "YES"}
    });
    ov::serialize(compiled_model.get_runtime_model(),
        "llm_exec_graph_"+std::to_string(3)+".xml");
    auto llm_infer_ = compiled_model.create_infer_request();
    auto type = ov::element::i64;
    std::cout << "Going to Iter" << std::endl;
    for (size_t count = 0; count < 2; count++) {
        size_t batch_size = is_beam_search ? 4 : 5;
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
        
        std::vector<int32_t> beam_idx(batch_size * 16, 0);
        std::vector<std::vector<int32_t> > next_beam_idx = {
            {0, 1, 2, 3}, {0, 0, 0, 0}, {2, 0, 3, 1}, {1, 0, 2, 3}, {0, 0, 1, 0}, 
            {2, 1, 3, 0}, {1, 0, 0, 3}, {2, 1, 0, 0}, {0, 1, 2, 3}, {0, 0, 2, 1}
        };
        std::iota(beam_idx.begin(), beam_idx.end(), 0);
        ov::Shape beam_idx_shape = {batch_size};
        ov::Tensor beam_idx_ov = ov::Tensor(ov::element::i32, beam_idx_shape, beam_idx.data());
        size_t max_new_length = is_beam_search ? next_beam_idx.size() : 10;
        for(size_t i = 0; i < 1; i++) {
            std::cout << "The " << i+1 << " token" << std::endl;
            if (i != 0) {
                attn_mask_ov = ov::Tensor(type, {batch_size, seq_len + i}, attn_mask.data() + batch_size * (seq_len + i));
                position_ids_ov = ov::Tensor(type, {batch_size, 1}, position_ids.data() + batch_size * (seq_len + i));
                input_ids_ov = ov::Tensor(type, {batch_size, 1}, input_ids.data());
                beam_idx_ov = is_beam_search ? ov::Tensor(ov::element::i32, beam_idx_shape, next_beam_idx[i].data())
                                             : ov::Tensor(ov::element::i32, beam_idx_shape, beam_idx.data());
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
                } else if (name == "beam_idx") {
                    llm_infer_.set_input_tensor(idx++, beam_idx_ov);
                }
            }

            try {
                const auto start = std::chrono::high_resolution_clock::now();
                llm_infer_.infer();
                const auto end = std::chrono::high_resolution_clock::now();
                const std::chrono::duration<double> diff = end - start;
                // skip print for warm-up
                if(count > 0)
                    std::cout << "Time to run token " << i << "|" << std::chrono::duration_cast<std::chrono::milliseconds>(diff).count() << std::endl;
            } catch(std::exception& ex) {
                std::cout << "infer" << ex.what() << std::endl;
            }
            std::cout << "Infer Finished" << std::endl;
            auto out = llm_infer_.get_output_tensor(0);
            std::cout << "output_shape|" << out.get_shape() << std::endl;
            // float* outPtr = out.data<float>();
            // for (size_t i = 0; i < 5; i++) {
            //     for(size_t j = 0; j < 5; j++) {
            //         std::cout << outPtr[i*4096 + j] << ","; 
            //     }
            //     std::cout << std::endl;
            // }
            // std::cout << std::endl;
        }
        // reset
        for (auto&& state : llm_infer_.query_state()) {
            state.reset();
        }
    }

    static int count = 0;
    ov::serialize(compiled_model.get_runtime_model(),
        "llm_exec_graph_"+std::to_string(count++)+".xml");
}


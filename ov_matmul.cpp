#include <iostream>
#include <openvino/opsets/opset8.hpp>
#include <openvino/pass/serialize.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>
#include "openvino/openvino.hpp"
#include <thread>
#include <stdlib.h>

using namespace ov;

int main(int args, char *argv[]) {

    ov::Core core;
    std::shared_ptr<ov::Model> model = core.read_model("/home/zhangyi7/wd_grouped.xml");
    // std::shared_ptr<ov::Model> model2 = core.read_model("/home/zhangyi7/ov_llm_sim/second_shared_matmul.xml");
    auto enc_inputs = model->inputs();
    auto ov_version = ov::get_openvino_version();
    std::cout << "OPENVINO|VERSION|" << ov_version << std::endl;
    // core.set_property("CPU", ov::num_streams(10));
    core.set_property("CPU", ov::inference_num_threads(1));
    // core.set_property("CPU", ov::affinity(ov::Affinity::NONE));
    auto precision = ov::element::f32;
    core.set_property("CPU", ov::hint::inference_precision(precision));
    core.set_property("CPU", ov::hint::dynamic_quantization_group_size(16));
    auto compiled1 = core.compile_model(model, "CPU", {
        {"PERF_COUNT", "YES"},
        {"NUM_STREAMS", "1"}
    });
    // auto compiled2 = core.compile_model(model2, "CPU", {
    //     {"PERF_COUNT", "YES"},
    //     {"NUM_STREAMS", "1"}
    // });
    std::cout << "compiled" << std::endl;
    auto infer_data = [](CompiledModel& exeNetwork) {
        auto encoder_infer_ = exeNetwork.create_infer_request();
        std::vector<float> input_data(8*16, 1);
        // std::vector<int32_t> input_int_data(32 * 32, 1);
        ov::Tensor input_ov = ov::Tensor(ov::element::f32, {1, 8, 16}, input_data.data());
        // ov::Tensor input2_int = ov::Tensor(ov::element::i32, {32, 32}, input_int_data.data());
        // ov::Tensor indice_ov = ov::Tensor(ov::element::i64, {17}, indice_data.data());
        encoder_infer_.set_input_tensor(0, input_ov);
        // encoder_infer_.set_input_tensor(0, input2_int);
        encoder_infer_.infer();
        auto output_ov = encoder_infer_.get_output_tensor(0);
        auto* p_data = output_ov.data<float>();
        std::cout << "Output data" << std::endl;
        for (size_t i = 0; i < 4; i++) {
            for(size_t j = 0; j < 16; j++) {
                std::cout << p_data[i * 16 + j] << ",";
            }
            std::cout << std::endl;
        }
        auto exec_graph = exeNetwork.get_runtime_model();
        serialize(exec_graph, "exec_matmul.xml");
        std::cout << std::endl;
    };
    std::cout << "Going to run first model!!!!!" << std::endl;
    infer_data(compiled1);
    // // setenv("DEBUG_PREPACK", "1", 0);
    // std::cout << "Going to run second model!!!!!" << std::endl;
    // infer_data(compiled2);
}

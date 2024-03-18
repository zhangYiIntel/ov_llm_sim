#include <iostream>
#include <algorithm>
#include <openvino/opsets/opset8.hpp>
#include <openvino/opsets/opset13.hpp>
#include <openvino/pass/serialize.hpp>
#include <openvino/core/node_vector.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>
#include "openvino/openvino.hpp"
#include "openvino/op/sink.hpp"
#include <thread>
#include <numeric>
#include <cassert>

using namespace ov;
using SinkVector = std::vector<std::shared_ptr<op::Sink>>;

template <typename IT, typename T>
void strided_iota(IT first, size_t n, T value, T stride) {
    for (size_t i = 0; i < n; i++) {
        *first++ = value;
        value += stride;
    }
}

std::shared_ptr<Model> get_sdpa_model(size_t seq_len, size_t head_size, size_t head_group_len, ov::element::Type qkvType, bool is_transpose = true) {
    ov::ParameterVector inputParams;
    size_t head_num = 8;
    // size_t head_group_len = 4;
    ov::PartialShape q_shape, kv_shape, past_shape;
    int64_t kv_head_num = static_cast<int64_t>(head_num / head_group_len);
    if (is_transpose) {
        // LBHS
        past_shape = {-1, 1, kv_head_num, static_cast<int64_t>(head_size)};
        q_shape = {-1, 1, static_cast<int64_t>(head_num), static_cast<int64_t>(head_size)};
        kv_shape = {-1, 1, kv_head_num, static_cast<int64_t>(head_size)};
    } else {
        // BHLS
        q_shape = {1, static_cast<int64_t>(head_num), -1, static_cast<int64_t>(head_size)};
        kv_shape = {1, kv_head_num, -1, static_cast<int64_t>(head_size)};
        past_shape = {1, kv_head_num, -1, static_cast<int64_t>(head_size)};
    }
    std::shared_ptr<ov::Node> q_in = nullptr;
    std::shared_ptr<ov::Node> k_in = nullptr;
    std::shared_ptr<ov::Node> v_in = nullptr;
    inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(qkvType, q_shape));
    inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(qkvType, kv_shape));
    inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(qkvType, kv_shape));
    inputParams[0]->set_friendly_name("q");
    inputParams[1]->set_friendly_name("k");
    inputParams[2]->set_friendly_name("v");
    // pastkv init_cost
    inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(qkvType, past_shape));
    inputParams[3]->set_friendly_name("past_kv");
    auto var_k = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{past_shape, qkvType, "pastk"});
    auto pastk = std::make_shared<ov::op::v6::ReadValue>(inputParams[3], var_k);
    pastk->set_friendly_name("pastk_r");
    auto var_v = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{past_shape, qkvType, "pastv"});
    auto pastv = std::make_shared<ov::op::v6::ReadValue>(inputParams[3], var_v);
    pastv->set_friendly_name("pastv_r");
    std::shared_ptr<Node> pastk_shapeof, pastv_shapeof;
    // auto transposeOrder = std::vector<size_t>{1, 2, 0, 3};
    std::vector<size_t> transposeOrder{1, 2, 0, 3};
    if (!is_transpose) {
        transposeOrder = std::vector<size_t>{0, 1, 2, 3};
    }
    // pre SDPA transpose
    auto preOrder = op::v0::Constant::create(ov::element::i32, {4}, transposeOrder);
    if (is_transpose) {
        q_in = std::make_shared<ov::op::v1::Transpose>(inputParams[0], preOrder);
    } else {
        q_in = inputParams[0];
    }

    auto concat_axis = transposeOrder[2];
    auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
    beam_idx->set_friendly_name("beam_idx");
    inputParams.push_back(beam_idx);
    auto gatherK = std::make_shared<ov::op::v8::Gather>(pastk, beam_idx, op::v0::Constant::create(ov::element::i32, {1}, {transposeOrder[0]}));
    auto gatherV = std::make_shared<ov::op::v8::Gather>(pastv, beam_idx, op::v0::Constant::create(ov::element::i32, {1}, {transposeOrder[0]}));
    auto concatK = std::make_shared<ov::op::v0::Concat>(OutputVector{gatherK, inputParams[1]}, concat_axis);
    auto concatV = std::make_shared<ov::op::v0::Concat>(OutputVector{gatherV, inputParams[2]}, concat_axis);

    if (head_group_len > 1) {
        size_t h_idx = is_transpose ? 2 : 1;
        auto unsquezeAxis = op::v0::Constant::create(ov::element::i32, {}, {h_idx + 1});
        auto unsqueezeK = std::make_shared<ov::op::v0::Unsqueeze>(concatK, unsquezeAxis);
        auto unsqueezeV = std::make_shared<ov::op::v0::Unsqueeze>(concatV, unsquezeAxis);
        ov::Shape shape = {1, 1, 1, 1, 1};
        shape[h_idx + 1] = head_group_len;
        auto targetShape = op::v0::Constant::create(qkvType, {1, 1, 1, head_group_len, 1}, {1});
        auto broadcastK = std::make_shared<ov::op::v1::Multiply>(unsqueezeK, targetShape);
        auto broadcastV = std::make_shared<ov::op::v1::Multiply>(unsqueezeV, targetShape);

        std::vector<size_t> targetShape2{0, 0, 0, head_size};
        targetShape2[h_idx] = head_num;
        auto target4D = op::v0::Constant::create(ov::element::i32, ov::Shape{4}, targetShape2);

        k_in = std::make_shared<ov::op::v1::Reshape>(broadcastK, target4D, true);
        v_in = std::make_shared<ov::op::v1::Reshape>(broadcastV, target4D, true);
    } else {
        k_in = concatK;
        v_in = concatV;
    }

    // auto transposeK = std::make_shared<ov::op::v1::Transpose>(k_in, preOrder);
    // auto transposeV = std::make_shared<ov::op::v1::Transpose>(v_in, preOrder);
    if (is_transpose) {
        k_in = std::make_shared<ov::op::v1::Transpose>(k_in, preOrder);
        v_in = std::make_shared<ov::op::v1::Transpose>(v_in, preOrder);
    }
    auto sdp = std::make_shared<ov::opset13::ScaledDotProductAttention>(q_in, k_in, v_in, false);
    sdp->set_friendly_name("mha");
    auto pastk_assign = std::make_shared<ov::op::v6::Assign>(concatK, var_k);
    auto pastv_assign = std::make_shared<ov::op::v6::Assign>(concatV, var_v);
    pastk_assign->set_friendly_name("pastk_w");
    pastv_assign->set_friendly_name("pastv_w");
        auto get_reshape_order = [](const ov::PartialShape& qkv_shape,
                                    const std::vector<size_t>& transposeOrder) -> std::vector<size_t> {
            assert(transposeOrder.size() == 4);
            auto H = qkv_shape[transposeOrder[1]].get_length();
            auto S = qkv_shape[transposeOrder[3]].get_length();
            return std::vector<size_t>{0, 0, static_cast<size_t>(H * S)};
        };
    // const auto reshapeOrder = get_reshape_order(q_shape, transposeOrder);

    // auto postOrder =
    //     ov::op::v0::Constant::create(ov::element::i32, {4}, std::vector<size_t>{2, 0, 1, 3});  // BHLS -> LBHS
    // auto transposeSDP = std::make_shared<ov::op::v1::Transpose>(sdp, postOrder);

    // auto constReshape = ov::op::v0::Constant::create(ov::element::i32, {3}, reshapeOrder);
    // auto reshapeSDP = std::make_shared<ov::op::v1::Reshape>(transposeSDP, constReshape, true);  // BLHS -> B,L,HxS

    // auto add = std::make_shared<ov::op::v1::Add>(reshapeSDP, op::v0::Constant::create(qkvType, {1}, {1.0f}));
    SinkVector sinks{pastk_assign, pastv_assign};
    ov::OutputVector results{sdp};
    auto model = std::make_shared<Model>(results, sinks, inputParams, "MultiQuery");
    return model;
}

template <typename T>
void model_infer(std::shared_ptr<ov::Model> model, size_t seq_len, size_t head_num, size_t head_size, size_t head_group_len, ov::element::Type type, bool is_transpose = true) {
    ov::Core core;
    core.set_property("CPU", ov::num_streams(1));
    core.set_property("CPU", ov::inference_num_threads(1));
    core.set_property("CPU", ov::affinity(ov::Affinity::NONE));
    core.set_property("CPU", ov::hint::inference_precision(type));
    auto compiled_model = core.compile_model(model, "CPU", {
        {"PERF_COUNT", "YES"}
    });

    ov::Shape q_shape, past_shape, kv_shape;
    if (is_transpose) {
        past_shape = ov::Shape{0, 1, head_num / head_group_len, head_size};
        q_shape = ov::Shape{seq_len, 1, head_num, head_size};
        kv_shape = ov::Shape{seq_len, 1, head_num / head_group_len, head_size};
    } else {
        // BHLS
        q_shape = ov::Shape{1, head_num, seq_len, head_size};
        kv_shape = ov::Shape{1, head_num / head_group_len, seq_len, head_size};
        past_shape = ov::Shape{1, head_num / head_group_len, 0, head_size};
    }
    auto llm_infer_ = compiled_model.create_infer_request();
    std::vector<T> q_data(seq_len * head_num * head_size, 4.0f);
    auto encoder_inputs_ = model->inputs();

    //input_cache
    // for (size_t i = 0; i < seq_len; i++) {
    //     for (size_t h = 0; h < head_num; h++) {
    //         // std::cout << "q h idx " << h << std::endl;
    //         for (size_t s = 0; s < head_size; s ++) {
    //             q_data[s + h * head_size + i * head_num * head_size] = h * 2.0f + 1.0f;
    //             // std::cout << q_data[s + h * head_size + i * 2 * head_size] << ",";
    //         }
    //         // std::cout << std::endl;
    //     }
    // }
    ov::Tensor q_tensor_ov = ov::Tensor(type, q_shape, q_data.data());

    // current k v
    std::vector<T> k_data(seq_len * head_num / head_group_len * head_size, 2.0f);
    // std::iota(k_data.begin(), k_data.end(), 0);
    // for (size_t i = 0; i < seq_len; i++) {
    //     for (size_t h = 0; h < head_num / head_group_len; h++) {
    //         // std::cout << "k h idx " << h << std::endl;
    //         for (size_t s = 0; s < head_size; s ++) {
    //             k_data[s + h * head_size + i * head_num / head_group_len * head_size] = h * 3.0f + 1.0f;
    //             // std::cout << k_data[s + h * head_size + i * 2 * head_size] << ",";
    //         }
    //         // std::cout << std::endl;
    //     }
    // }

    ov::Tensor k_tensor_ov = ov::Tensor(type, kv_shape, k_data.data());
    std::vector<T> v_data(seq_len * head_num / head_group_len * head_size, 2);
    // for (size_t i = 0; i < seq_len; i++) {
    //     for (size_t h = 0; h < head_num / head_group_len; h++) {
    //         // std::cout << "k h idx " << h << std::endl;
    //         for (size_t s = 0; s < head_size; s ++) {
    //             v_data[s + h * head_size + i * head_num / head_group_len * head_size] = h * 2.0f + 1.0f;
    //             // std::cout << v_data[s + h * head_size + i * 2 * head_size] << ",";
    //         }
    //         // std::cout << std::endl;
    //     }
    // }
    ov::Tensor v_tensor_ov = ov::Tensor(type, kv_shape, v_data.data());

    // past kv
    std::vector<T> past_kv_data(1, 1);
    ov::Tensor past_kv_tensor_ov = ov::Tensor(type, past_shape, past_kv_data.data());

    std::vector<int32_t> beam_idx = {0};
    ov::Shape beam_idx_shape = {1};
    ov::Tensor beam_idx_ov = ov::Tensor(ov::element::i32, beam_idx_shape, beam_idx.data());
    // fake hidden stat

    std::cout << "Going to Iter" << std::endl;

    for(size_t i = 0; i < 1; i++) {
        std::cout << "The " << i+1 << " token" << std::endl;
        int idx = 0;
        for (auto& input : encoder_inputs_) {
            std::string name = input.get_node()->get_friendly_name();
            std::cout << "set input " << name << std::endl;
            if (name == "q") {
                llm_infer_.set_input_tensor(idx++, q_tensor_ov);
            } else if (name == "k") {
                llm_infer_.set_input_tensor(idx++, k_tensor_ov);
            } else if (name == "v") {
                llm_infer_.set_input_tensor(idx++, v_tensor_ov);
            } else if (name == "past_kv") {
                llm_infer_.set_input_tensor(idx++, past_kv_tensor_ov);
            } else if (name == "beam_idx") {
                llm_infer_.set_input_tensor(idx++, beam_idx_ov);
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
        auto out_shape = out.get_shape();
        auto outPtr = out.data<T>();
        static int count = 0;
        ov::serialize(compiled_model.get_runtime_model(),
            "simple_exec_graph_"+std::to_string(count++)+".xml");
        // if (out_shape.size() == 4) {
        //     for (size_t seq_len = 0;  seq_len < out_shape[0]; seq_len ++) {
        //         std::cout << "seq " << seq_len << std::endl;
        //         for (size_t head_idx = 0; head_idx < out_shape[2]; head_idx ++) {
        //             for (size_t i = 0; i < out_shape[3]; i ++) {
        //                 std::cout << outPtr[i + head_idx * out_shape[3] + seq_len * out_shape[2] * out_shape[3]] << ",";
        //             }
        //             std::cout << std::endl;
        //         }
        //         std::cout << std::endl;
        //     }
        // }

    }
}

int main(int args, char *argv[]) {
    if (args < 2)
        exit(-1);
    std::string length(argv[1]);
    size_t seq_len = std::stoi(length);
    auto type = getenv("DISABLE_BF16") ? ov::element::f32 : ov::element::bf16;
    size_t head_num = 8;
    size_t head_size = 32;
    size_t head_group_len = 4;
    bool is_transpose = args >=3 ? std::string(argv[2]) == "T" ? true : false : true;
    std::shared_ptr<ov::Model> model = get_sdpa_model(seq_len, head_size, head_group_len, type, is_transpose);
    std::cout << "model creation success!" << std::endl;
    auto ov_version = ov::get_openvino_version();
    // ov::serialize(model,
    //     "simple_sdpa.xml");
    std::cout << "OPENVINO|VERSION|" << ov_version << std::endl;
    if (type == ov::element::bf16) {
        std::cout << "Run BF16" << std::endl;
        model_infer<ov::bfloat16>(model, seq_len, head_num, head_size, head_group_len, type, is_transpose);
    } else {
        std::cout << "Run FP32" << std::endl;
        model_infer<float>(model, seq_len, head_num, head_size, head_group_len, type, is_transpose);
    }
}

cmake_minimum_required(VERSION 3.8)
project(llm_test)

set(CMAKE_CXX_STANDARD 11)
find_package(OpenVINO REQUIRED)

add_executable(llm_test main.cpp)
target_link_libraries(llm_test openvino::runtime)

add_executable(simple_sdpa simple_sdpa.cpp)
target_link_libraries(simple_sdpa openvino::runtime)

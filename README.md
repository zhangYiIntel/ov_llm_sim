# LLM Simulator for OpenVINO
This repo contains miminal code to run LLM models with OpenVINO C++ API. It aims to fasilicate the development of kernels. It only keeps the inference logic and omits the word search in llm. 

# Usage
## Build
Make sure you have openvino installed and you have source the `setupvars.sh`, otherwise you need to set -D
OpenVINO_DIR=<your ov build dir> for cmake configuration
```
mkdir build
cmake ..
make
```
## Run
llm_test
```
llm_test <path to model>
```
simple_sdpa
```
simple_sdpa <1st token length> [T/N]
```

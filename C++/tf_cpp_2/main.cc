#include "tensorflow.h"

#include <iostream>
#include <string>
#include <vector>

int main() {
    
    using namespace tensorflow;

    SavedModelBundle bundle;

    std::string import_path = {"E:\\dev_libraries\\tensorflow\\tensorflow\\tf_cpp_2\\trained_model"};

    SessionOptions session_options;
    RunOptions run_options;
    std::unordered_set<std::string> tags = {"train"}; //kSavedModelTagsTrain

    LoadSavedModel(session_options, run_options, import_path, tags, &bundle);

    tensorflow::Tensor a(DT_FLOAT, TensorShape({1}));
    auto a_map = a.tensor<float, 1>();
    a_map(0) = 2.0;
      
    std::vector<tensorflow::Tensor> outputs;

    TF_CHECK_OK(bundle.session->Run({ {"a", a} }, {"b"}, {}, &outputs));

    LOG(INFO) << outputs[0].DebugString();

    auto tensor_map = outputs[0].tensor<float, 1>();
    auto array = tensor_map.data();
    float* float_array = static_cast<float*>(array);

    std::cout << float_array[0] << std::endl;

    return 0;

}
#include "tensorflow.h"

#include <vector>
#include <string>

int main() {
    
    using namespace tensorflow;
    
    Scope root = Scope::NewRootScope();

    auto a = ops::Placeholder(root, DT_FLOAT);
    auto b = ops::Const(root, {3.f});

    auto s = ops::Add(root, a, b);

    std::vector<Tensor> outputs;

    ClientSession session(root);

    TF_CHECK_OK(session.Run({ {a, {1.f}} }, {s}, &outputs));

    LOG(INFO) << outputs[0].scalar<float>();

    //Image

    Scope file_root = Scope::NewRootScope();

    auto f_in = ops::Placeholder(file_root, DT_STRING);
  
    auto reader = ops::ReadFile(file_root, f_in);

    auto decoded = ops::DecodePng(file_root, reader, tensorflow::ops::DecodePng::Channels(3));

    auto shape = ops::Shape(file_root, decoded);

    ClientSession file_session(file_root);

    std::string input_file = {"E:\\dev_libraries\\tensorflow\\bazel-bin\\tensorflow\\tf_cpp_1\\test_image.png"};
    
    TF_CHECK_OK(file_session.Run({ {f_in, input_file} }, {shape}, &outputs));

    LOG(INFO) << outputs[0].DebugString();
       
    return 0;

}
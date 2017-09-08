#include <iostream>
#include <string>
#include <vector>

#include <opencv2\opencv.hpp>

#include "tensorflow.h"

int main() {
    
    using namespace tensorflow;

    SavedModelBundle bundle;
    
    std::string import_path = {"E:\\dev_libraries\\tensorflow\\tensorflow\\tf_cpp_4\\trained_model"};
    // std::string import_path = {"E:\\dev_libraries\\tensorflow\\tensorflow\\tf_cpp_4\\trained_model_test"};
    
    SessionOptions session_options;
    RunOptions run_options;
    std::unordered_set<std::string> tags = {"train"}; //kSavedModelTagsTrain
    
    LoadSavedModel(session_options, run_options, import_path, tags, &bundle);
 
    cv::Mat cv_in = cv::imread("E:\\dev_libraries\\tensorflow\\tensorflow\\tf_cpp_4\\feature.png");

    int rows = cv_in.rows;
    int cols = cv_in.cols;
    int nchannels = 3;

    cv::imshow("cv_in", cv_in);

    cv::cvtColor(cv_in, cv_in, CV_BGR2RGB);

    cv_in.convertTo(cv_in, CV_32FC3);  

    Tensor image(DT_FLOAT, TensorShape({1, rows, cols, 3}));
    auto tensor_map = image.tensor<float, 4>();

    auto source_data = reinterpret_cast<float*>(cv_in.data);  

    for(int r=0; r<rows; ++r){
        for(int c=0; c<cols; ++c){
            for(int ch=0; ch<nchannels; ++ch){
                tensor_map(0, r, c, ch) = source_data[(nchannels*cols*r) + (nchannels*c) + ch];
            }
        }
    }
       
    std::vector<Tensor> outputs;

    TF_CHECK_OK(bundle.session->Run({ {"input/image:0", image} }, {"export/prediction:0"}, {}, &outputs));

    LOG(INFO) << outputs[0].DebugString();

    auto b = outputs[0].tensor<uint8, 4>();
    auto array = b.data();

    auto out_data = reinterpret_cast<uint8*>(array);

    cv::Mat cv_out(cv::Size(cols, rows), CV_8UC1, out_data);//CV_32FC1, out_data);

    cv_out.convertTo(cv_out, CV_8UC1, 255, 0);
    cv::imshow("cv_out", cv_out);
    
    cv::waitKey(0);

    return 0;

}
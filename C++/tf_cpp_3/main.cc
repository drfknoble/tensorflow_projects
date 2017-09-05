#include "tensorflow.h"

#include <iostream>
#include <string>
#include <vector>

#include <opencv2\opencv.hpp>

int main() {
    
    using namespace tensorflow;

    SavedModelBundle bundle;
    
    std::string import_path = {"E:\\dev_libraries\\tensorflow\\tensorflow\\tf_cpp_3\\trained_model"};
    
    SessionOptions session_options;
    RunOptions run_options;
    std::unordered_set<std::string> tags = {"train"}; //kSavedModelTagsTrain
    
    LoadSavedModel(session_options, run_options, import_path, tags, &bundle);

    std::vector<tensorflow::Tensor> outputs;

    cv::Mat cv_image;

    cv_image = cv::imread("E:\\dev_libraries\\tensorflow\\tensorflow\\tf_cpp_3\\test_image.png");

    if(cv_image.empty()) {
        std::cout << "Could not open image." << std::endl;
        return 1;
    }

    cv::namedWindow("Input Image", CV_WINDOW_AUTOSIZE);

    cv::imshow("Input Image", cv_image);
    
    cv::cvtColor(cv_image, cv_image, CV_BGR2RGB);
   
    Tensor a(DT_UINT8, TensorShape({160, 120, 3}));
    auto tensor_map = a.tensor<uint8, 3>();

    auto source_data = (uint8*) cv_image.data;
    
     for (int y = 0; y < 160; ++y) {
        const uint8* source_row = source_data + (y * 120 * 3);
        for (int x = 0; x < 120; ++x) {
            const uint8* source_pixel = source_row + (x * 3);
            for (int c = 0; c < 3; ++c) {
                const uint8* source_value = source_pixel + c;
                tensor_map(y, x, c) = *source_value;
        }
      }
    }  

    TF_CHECK_OK(bundle.session->Run({ {"a", a} }, {"b"}, {}, &outputs));
    
    LOG(INFO) << outputs[0].DebugString();

    auto b = outputs[0].tensor<uint8, 3>();
    auto array = b.data();
    uint8* int_array = static_cast<uint8*>(array);

    cv::Mat tf_image(cv::Size(160, 120), CV_8UC3, int_array);

    cv::cvtColor(tf_image, tf_image, CV_RGB2BGR);

    cv::namedWindow("TensorFlow Image", CV_WINDOW_AUTOSIZE);

    cv::imshow("TensorFlow Image", tf_image);
    
    cv::waitKey(0);

    cv::destroyAllWindows();

    return 0;

}
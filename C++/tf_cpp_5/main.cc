#include <iostream>
#include <string>
#include <vector>

#include <opencv2\opencv.hpp>

#include "tensorflow.h"

int main() {
    
    using namespace tensorflow;

    SavedModelBundle bundle;
    
    std::string import_path = {"E:\\dev_libraries\\tensorflow\\tensorflow\\tf_cpp_5\\trained_model"};
    
    SessionOptions session_options;
    RunOptions run_options;
    std::unordered_set<std::string> tags = {"train"}; //kSavedModelTagsTrain
    
    LoadSavedModel(session_options, run_options, import_path, tags, &bundle);

    cv::VideoCapture camera(0);
    
    if(!camera.isOpened()) {
        std::cout << "Could not open camera." << std::endl;
        return 1;
    }

    cv::namedWindow("Frame", CV_WINDOW_AUTOSIZE);
    cv::namedWindow("Processed Frame", CV_WINDOW_AUTOSIZE);


    while (true) {
      
        cv::Mat frame;
    
        camera >> frame;

        if (!frame.empty()) {

            cv::imshow("Frame", frame);

            cv::resize(frame, frame, cv::Size(160, 120));            
            
            int rows = frame.rows;
            int cols = frame.cols;
            int nchannels = 3;
                   
            cv::cvtColor(frame, frame, CV_BGR2RGB);
                    
            Tensor image(DT_UINT8, TensorShape({1, rows, cols, 3}));
            auto tensor_map = image.tensor<uint8, 4>();
            
            auto source_data = reinterpret_cast<uint8*>(frame.data);  
            
            for(int r=0; r<rows; ++r){
                for(int c=0; c<cols; ++c){
                    for(int ch=0; ch<nchannels; ++ch){
                            tensor_map(0, r, c, ch) = source_data[(nchannels*cols*r) + (nchannels*c) + ch];
                    }
                }
            }

            std::vector<Tensor> outputs;    
                   
            TF_CHECK_OK(bundle.session->Run({ {"input/image:0", image} }, {"export/predicted:0"}, {}, &outputs));
                        
            auto b = outputs[0].tensor<uint8, 4>();
            auto array = b.data();
            
            auto out_data = reinterpret_cast<uint8*>(array);
            
            cv::Mat processed_frame(cv::Size(cols, rows), CV_8UC1, out_data);
            
            processed_frame.convertTo(processed_frame, CV_8UC1, 255, 0);

            cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(5, 5), cv::Point(2, 2));
            
            cv::erode(processed_frame, processed_frame, element);
            cv::erode(processed_frame, processed_frame, element);
            cv::dilate(processed_frame, processed_frame, element);
            cv::dilate(processed_frame, processed_frame, element);

            cv::resize(processed_frame, processed_frame, cv::Size(640, 480));            
            
            cv::imshow("Processed Frame", processed_frame);

        }

        char c;
        c = cv::waitKey(1);
        if(c == 27) {
            break;
        }

    }

    cv::destroyAllWindows();

    camera.release();

    return 0;

}
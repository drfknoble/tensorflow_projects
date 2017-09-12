#include <iostream>
#include <string>
#include <vector>

#include <opencv2\opencv.hpp>

#include "tensorflow.h"

int main() {
    
    using namespace tensorflow;
   
    std::string import_file = {"E:\\dev_libraries\\tensorflow\\tensorflow\\opencv_tensorflow\\trained_model\\frozen_inference_graph.pb"};

    Session* session;
    GraphDef graph_def;
    
    TF_CHECK_OK(NewSession(SessionOptions(), &session));
    
    TF_CHECK_OK(ReadBinaryProto(Env::Default(), import_file, &graph_def));

    TF_CHECK_OK(session->Create(graph_def));
      
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

            cv::resize(frame, frame, cv::Size(640, 480));            
            
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
                   
            TF_CHECK_OK(session->Run({ {"image_tensor:0", image} }, {"detection_boxes:0"}, {}, &outputs));

            auto b = outputs[0].tensor<float, 3>();
            auto array = b.data();
            
            auto out_data = reinterpret_cast<float*>(array);
            
            cv::Mat processed_frame = frame.clone();        
            
            cv::cvtColor(processed_frame, processed_frame, CV_RGB2BGR);

            cv::resize(processed_frame, processed_frame, cv::Size(640, 480));
            
            int x_min = 0, y_min = 0, x_max = 0, y_max = 0;
            
            for (int i = 0; i < 5; i++) {

                y_min = array[4*i]*frame.rows;
                x_min = array[4*i+1]*frame.cols;
                y_max = array[4*i+2]*frame.rows;
                x_max = array[4*i+3]*frame.cols;
    
                cv::rectangle(processed_frame, cv::Point(x_min, y_min), cv::Point(x_max, y_max), cv::Scalar(0, 0, 255), 4);
  
            }

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
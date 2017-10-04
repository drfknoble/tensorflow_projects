#include <iostream>
#include <sstream>
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
                   
            TF_CHECK_OK(session->Run({ {"image_tensor:0", image} }, {"detection_boxes:0", "detection_scores:0", "detection_classes:0", "num_detections:0"}, {}, &outputs));

            // LOG(INFO) << "Boxes: " << outputs[0].DebugString();
            // LOG(INFO) << "Scores: " << outputs[1].DebugString();
            // LOG(INFO) << "Classes: " << outputs[2].DebugString();
            // LOG(INFO) << "Detected Objects: " << outputs[3].DebugString();
            
            auto boxes = outputs[0].tensor<float, 3>();
            auto boxes_array = boxes.data();
            auto boxes_data = reinterpret_cast<float*>(boxes_array);

            auto scores = outputs[1].tensor<float, 2>();
            auto scores_array = scores.data();
            auto scores_data = reinterpret_cast<float*>(scores_array);

            auto classes = outputs[2].tensor<float, 2>();
            auto classes_array = classes.data();
            auto classes_data = reinterpret_cast<float*>(classes_array);

            auto num = outputs[3].tensor<float, 1>();
            auto num_array = num.data();
            auto num_data = reinterpret_cast<float*>(num_array);
            
            cv::Mat processed_frame = frame.clone();        
            
            cv::cvtColor(processed_frame, processed_frame, CV_RGB2BGR);

            cv::resize(processed_frame, processed_frame, cv::Size(640, 480));
            
            int x_min = 0, y_min = 0, x_max = 0, y_max = 0;
            
            for (int i = 0; i < static_cast<int>(num_data[0]); i++) {

                if (classes_data[i] == 1) {

                    if(scores_data[i] > 0.80) {

                        y_min = boxes_data[4*i]*frame.rows;
                        x_min = boxes_data[4*i+1]*frame.cols;
                        y_max = boxes_data[4*i+2]*frame.rows;
                        x_max = boxes_data[4*i+3]*frame.cols;
                    
                        std::stringstream label_stream;
                        label_stream << classes_data[i] << ": " << 100*scores_data[i];
        
                        std::string label = label_stream.str();
                    
                        cv::putText(processed_frame, label, cv::Point(x_min, y_min), cv::HersheyFonts::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0,0,255), 1);
                        cv::rectangle(processed_frame, cv::Point(x_min, y_min), cv::Point(x_max, y_max), cv::Scalar(0, 0, 255), 4);

                    }
    
                }
  
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
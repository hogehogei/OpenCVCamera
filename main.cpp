#include <iostream>
#include <thread>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

#include "SurveillanceCamera.hpp"

int main( int argc, char** argv ) 
{

    constexpr float score_threshold = 0.95;
    constexpr float nms_threshold = 0.3;
    constexpr float topK = 5000;
    FaceDetector::Setting setting = {
        "model/face_detection_yunet_2022mar_int8.onnx",    // Model filepath
        0,                                                 // Image Width(Zero=SameCameraCaptureSize)
        0,                                                 // Image Height(Zero=SameCameraCaptureSize)
        score_threshold,
        nms_threshold,
        topK
    };

    //cv::setNumThreads(0);

    std::shared_ptr<SurveillanceCamera> camera = std::make_shared<SurveillanceCamera>( setting );
    if( camera->GetState() == SurveillanceCamera::ERROR_OPEN_RECORDER ){
        std::cerr << "Failed open recorder." << std::endl;
        return 1;
    }

    while(1){
        if( camera->GetState() == SurveillanceCamera::ERROR_RECORDER ){
            std::cerr << "Recording error happened." << std::endl;
            break;
        }
        camera->Update();
        
        //std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    return 0;
}
#ifndef SURVEILLANCE_HPP_INCLUDED
#define SURVEILLANCE_HPP_INCLUDED

#include <cstdint>
#include <thread>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>

#include "Mutex.hpp"



class FaceDetector
{
public:
    struct Setting
    {
        std::string ModelFilePath;
        uint32_t    Width;
        uint32_t    Height;
        float       ScoreThreshold;
        float       NMSThreshold;
        float       TopK;
    };
    enum State
    {
        IDLE,
        OPENED,
        ERROR_FAIL_INITIALIZE,
        ERROR_FAIL_START,
        ERROR_DETECT_THREAD,
        FACE_DETECTING,
        FACE_DETECT_OK,
        FACE_DETECT_NO_FACE
    };

public:

    static const int sk_VisualizeBorderThikness = 2; 

    FaceDetector();
    ~FaceDetector();
    FaceDetector( const FaceDetector& ) = delete;
    FaceDetector& operator=( const FaceDetector& ) = delete;

    bool Open( const FaceDetector::Setting& setting );
    State Detect( cv::Mat image );
    State DetectResult() const;
    void WaitDetectResult();
    cv::Mat GetFaceDetectVisualizedImage() const;

private:

    void DetectThread();

    FaceDetector::Setting        m_Setting;
    cv::Ptr<cv::FaceDetectorYN>  m_FaceDetector;
    std::unique_ptr<std::thread> m_FaceDetectThread;

    cv::Mat m_Image;
    cv::Mat m_Faces;

    MutexGuard<State> m_State;
};

class ImageWriter
{
public:

    static constexpr int sk_QueueMaxSize = 5;

    ImageWriter( cv::VideoWriter writer );
    ~ImageWriter();
    ImageWriter( const ImageWriter& ) = delete;
    ImageWriter& operator=( const ImageWriter& ) = delete;

    void Start();
    bool Enqueue( cv::Mat image );
    void End();
    bool IsError() const;

private:

    void WriterThread();

    MutexGuard<bool>                m_IsUsed;
    std::unique_ptr<std::thread>    m_ImgWriteThread;
    MutexGuard<bool>                m_ImageWriteStart;

    cv::VideoWriter  m_Writer;
    MutexGuard<bool> m_IsError;

    std::queue<cv::Mat> m_WriteQueue;
    std::mutex m_QueueLock;
};

class SurveillanceCamera
{
public:

    // 連続でエラーが発生した場合のエラー判定回数
    static constexpr int sk_RecorderConsecutiveErrorThreshold = 3;
    // 顔判定がなくなった時に、録画停止するまでの顔判定無し判定回数
    // 設定した回数連続で顔判定無しの場合は録画停止
    static constexpr int sk_NoDetectFaceThreshold = 5;

    enum State
    {
        INITIALIZING,
        STREAMING,
        STREAMING_AND_RECORDING_FACES,
        ERROR_OPEN_RECORDER,
        ERROR_RECORDER
    };

    SurveillanceCamera( const FaceDetector::Setting& setting );
    ~SurveillanceCamera();
    SurveillanceCamera( const SurveillanceCamera& ) = delete;
    SurveillanceCamera& operator=( const SurveillanceCamera& ) = delete;

    void Update();
    State GetState();

private:

    void ChangeSeqInitializing();
    
    void DoStreaming();
    FaceDetector::State DetectFace( cv::Mat frame );
    bool CreateDetectedFaceRecorder();
    void ChangeSeqStreaming();

    void DoStreamingAndRecordingFaces();
    void ChangeSeqStreamingAndRecordingFaces();

    bool IsError() const;


    State        m_CameraState;
    cv::VideoCapture m_Capture;
    uint32_t m_RecorderConsecutiveErrorCount;
    
    FaceDetector m_Detector;
    FaceDetector::Setting m_DetectorSetting;
    FaceDetector::State m_DetectState;
    FaceDetector::State m_PrevDetectState;
    uint32_t m_NoDetectFaceTime;
    cv::Mat  m_Faces;

    std::shared_ptr<ImageWriter>  m_DetectedFaceRecorder;
    std::shared_ptr<ImageWriter>  m_WebStreamWriter;
};

#endif  // SURVEILLANCE_HPP_INCLUDED
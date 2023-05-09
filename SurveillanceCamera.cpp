
#include "SurveillanceCamera.hpp"

#include <iomanip>
#include <sstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

namespace {

std::string BuildTimeStampString() {
    time_t t = time(nullptr);
    const tm* localTime = localtime(&t);

    std::stringstream s;
    s << localTime->tm_year + 1900;
    // setw(),setfill()で0詰め
    s << std::setw(2) << std::setfill('0') << localTime->tm_mon + 1;
    s << std::setw(2) << std::setfill('0') << localTime->tm_mday;
    s << std::setw(2) << std::setfill('0') << localTime->tm_hour;
    s << std::setw(2) << std::setfill('0') << localTime->tm_min;
    s << std::setw(2) << std::setfill('0') << localTime->tm_sec;

    return s.str();
}

void PrintDetectState( FaceDetector::State state )
{
    switch( state ){
    case FaceDetector::IDLE:
        std::cout << "FaceDetectorState: IDLE" << std::endl;
        break;
    case FaceDetector::OPENED:
        std::cout << "FaceDetectorState: OPENED" << std::endl;
        break;
    case FaceDetector::ERROR_FAIL_INITIALIZE:
        std::cout << "FaceDetectorState: ERROR_FAIL_INITIALIZE" << std::endl;
        break;
    case FaceDetector::ERROR_FAIL_START:
        std::cout << "FaceDetectorState: ERROR_FAIL_START" << std::endl;
        break;
    case FaceDetector::ERROR_DETECT_THREAD:
        std::cout << "FaceDetectorState: ERROR_DETECT_THREAD" << std::endl;
        break;
    case FaceDetector::FACE_DETECTING:
        std::cout << "FaceDetectorState: FACE_DETECTING" << std::endl;
        break;
    case FaceDetector::FACE_DETECT_OK:
        std::cout << "FaceDetectorState: FACE_DETECT_OK" << std::endl;
        break;
    case FaceDetector::FACE_DETECT_NO_FACE:
        std::cout << "FaceDetectorState: FACE_DETECT_NO_FACE" << std::endl;
        break;
    }
}

}

FaceDetector::FaceDetector()
    :
      m_Setting(),
      m_FaceDetector(),
      m_FaceDetectThread(),
      m_Image(),
      m_Faces(),
      m_State( FaceDetector::IDLE )
{}

FaceDetector::~FaceDetector()
{
    WaitDetectResult();
}

bool FaceDetector::Open( const FaceDetector::Setting& setting )
{
    if( m_State.Value != FaceDetector::IDLE ){
        return false;
    }

    m_Setting = setting;

    try {
        m_FaceDetector = cv::FaceDetectorYN::create(
            m_Setting.ModelFilePath,
            "",
            { m_Setting.Width, m_Setting.Height },
            m_Setting.ScoreThreshold,
            m_Setting.NMSThreshold,
            m_Setting.TopK
        );
        m_State.Value = FaceDetector::OPENED;
    }
    catch( cv::Exception& e ){
        std::cerr << e.what() << std::endl;
        m_State.Value = FaceDetector::ERROR_FAIL_INITIALIZE;
        return false;
    }

    return true;
}

FaceDetector::State FaceDetector::Detect( cv::Mat image )
{
    WaitDetectResult();

    std::lock_guard<std::mutex> guard( m_State.Mutex );
    if( m_State.Value == FaceDetector::ERROR_FAIL_INITIALIZE ){
        return FaceDetector::ERROR_FAIL_INITIALIZE;
    }

    try {
        m_Image = image;
        m_State.Value = FaceDetector::FACE_DETECTING;
        m_FaceDetectThread = std::make_unique<std::thread>( &FaceDetector::DetectThread, this );
    }
    catch( std::system_error& e ){
        std::cerr << e.what() << std::endl;
        m_State.Value = FaceDetector::ERROR_FAIL_START;
        return FaceDetector::ERROR_FAIL_START;
    }

    // もし万が一、ここに来るまでにスレッド処理が終わっていたとしても
    // この関数自体は DETECTING を返すこととする。
    return FaceDetector::FACE_DETECTING;
}

FaceDetector::State FaceDetector::DetectResult() const
{
    return m_State.Get();
}

void FaceDetector::WaitDetectResult()
{
    if( m_FaceDetectThread.get() && m_FaceDetectThread->joinable() ){
        m_FaceDetectThread->join();
    }
}

cv::Mat FaceDetector::GetFaceDetectVisualizedImage() const
{
    std::lock_guard<std::mutex> guard( m_State.Mutex );
    
    if( m_State.Value != FaceDetector::FACE_DETECT_OK ){
        return cv::Mat();
    }
    return m_Image;
}

void FaceDetector::DetectThread()
{
    constexpr int thickness = sk_VisualizeBorderThikness;

    try {
        m_FaceDetector->detect( m_Image, m_Faces );

        for( int i = 0; i < m_Faces.rows; ++i ){
            // Print results
            std::cout << "Face " << i
                << ", top-left coordinates: (" << m_Faces.at<float>(i, 0) << ", " << m_Faces.at<float>(i, 1) << "), "
                << "box width: " << m_Faces.at<float>(i, 2)  << ", box height: " << m_Faces.at<float>(i, 3) << ", "
                << "score: " << cv::format("%.2f", m_Faces.at<float>(i, 14))
                << std::endl;

            // Draw bounding box
            cv::rectangle( 
                m_Image,
                cv::Rect2i(
                    static_cast<int>(m_Faces.at<float>(i, 0)), 
                    static_cast<int>(m_Faces.at<float>(i, 1)), 
                    static_cast<int>(m_Faces.at<float>(i, 2)), 
                    static_cast<int>(m_Faces.at<float>(i, 3))
                ), 
                cv::Scalar(0, 255, 0), 
                thickness
            );

            // Draw landmarks
            cv::circle( m_Image, cv::Point2i(int(m_Faces.at<float>(i, 4)), int(m_Faces.at<float>(i, 5))), 2, cv::Scalar(255, 0, 0), thickness );
            cv::circle( m_Image, cv::Point2i(int(m_Faces.at<float>(i, 6)), int(m_Faces.at<float>(i, 7))), 2, cv::Scalar(0, 0, 255), thickness );
            cv::circle( m_Image, cv::Point2i(int(m_Faces.at<float>(i, 8)), int(m_Faces.at<float>(i, 9))), 2, cv::Scalar(0, 255, 0), thickness );
            cv::circle( m_Image, cv::Point2i(int(m_Faces.at<float>(i, 10)), int(m_Faces.at<float>(i, 11))), 2, cv::Scalar(255, 0, 255), thickness );
            cv::circle( m_Image, cv::Point2i(int(m_Faces.at<float>(i, 12)), int(m_Faces.at<float>(i, 13))), 2, cv::Scalar(0, 255, 255), thickness );
        }

        if( m_Faces.rows < 1 ){
            m_State.Set( FACE_DETECT_NO_FACE );
            std::cout << "NOFACE" << std::endl;
        }
        else {
            m_State.Set( FACE_DETECT_OK );
            std::cout << "DETECT OK" << std::endl;
        }
    }
    // 例外をすべてキャッチして、今後の書き込みを禁止する。
    // 例外をキャッチしないと親スレッドごと落ちてしまうため。
    // 必要であればエラーコード設定処理を追加。
    // ログ書き込みやロック程度でも例外送出されるなら落ちてもしょうがない
    catch( cv::Exception& e ){
        std::cerr << e.what() << std::endl;
        m_State.Set( ERROR_DETECT_THREAD );
    }
    catch( ... ){
        m_State.Set( ERROR_DETECT_THREAD );
    }
}


ImageWriter::ImageWriter( cv::VideoWriter writer )
    : 
      m_IsUsed( false ),
      m_ImgWriteThread(),
      m_ImageWriteStart( false ),
      m_Writer( writer ),
      m_IsError( false ),
      m_WriteQueue(),
      m_QueueLock()
{}

ImageWriter::~ImageWriter()
{
    m_ImageWriteStart.Set( false );
    if( m_ImgWriteThread.get() && m_ImgWriteThread->joinable() ){
        m_ImgWriteThread->join();
    }
}

void ImageWriter::Start()
{
    std::lock_guard<std::mutex> guard( m_IsUsed.Mutex );
    try {
        if( m_IsUsed.Value == false ){
            m_IsUsed.Value = true;
            m_ImageWriteStart.Set( true );
            // メンバ関数を引数に実行
            m_ImgWriteThread = std::make_unique<std::thread>(&ImageWriter::WriterThread, this);
        }
    }
    catch( ... ){
        m_ImageWriteStart.Set( false );
        m_IsError.Set( true );
    }
}

bool ImageWriter::Enqueue( cv::Mat image )
{
    if( m_ImageWriteStart.Get() == false ){
        std::cerr << "Enqueue failed" << std::endl;
        return false;
    }

    std::lock_guard<std::mutex> guard( m_QueueLock );
    if( m_WriteQueue.size() >= sk_QueueMaxSize ){
        std::cerr << "Can't enqueue because queue full." << std::endl;
        return false;
    }

    std::cerr << "Queued image file to ImageWriter" << std::endl;
    m_WriteQueue.push( image );

    return true;
}

void ImageWriter::End()
{
    std::lock_guard<std::mutex> guard( m_IsUsed.Mutex );
    if( m_IsUsed.Value == true ){
        m_ImageWriteStart.Set( false );
        m_ImgWriteThread->join();
    }
}

bool ImageWriter::IsError() const
{
    return m_IsError.Get();
}

void ImageWriter::WriterThread()
{
    try {
        while( m_ImageWriteStart.Get() == true )
        {
            bool is_empty = false;
            {
                std::lock_guard<std::mutex> guard( m_QueueLock );
                is_empty = m_WriteQueue.empty();
            }

            if( !is_empty ){
                cv::Mat image;
                {
                    std::lock_guard<std::mutex> guard( m_QueueLock );
                    image = m_WriteQueue.front();
                    m_WriteQueue.pop();
                }

                m_Writer << image;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    // 例外をすべてキャッチして、今後の書き込みを禁止する。
    // 例外をキャッチしないと親スレッドごと落ちてしまうため。
    // 必要であればエラーコード設定処理を追加。
    // ログ書き込みやロック程度でも例外送出されるなら落ちてもしょうがない
    catch( cv::Exception& e ){
        std::cerr << e.what() << std::endl;
        m_ImageWriteStart.Set( false );
        m_IsError.Set( true );
    }
    catch( ... ){
        std::cerr << "WriteImage thread aborted." << std::endl;
        m_ImageWriteStart.Set( false );
        m_IsError.Set( true );
    }

    m_Writer.release();
}

SurveillanceCamera::SurveillanceCamera( const FaceDetector::Setting& setting )
    :
    m_CameraState( SurveillanceCamera::INITIALIZING ),
    // cv::VideoCapture.set() では設定できなかったので、
    // gstreamer のパイプラインから指定
    m_Capture( "v4l2src device=/dev/video0 ! image/jpeg,width=1280, height=720, framerate=(fraction)30/1 !jpegdec !videoconvert ! appsink max-buffers=1 drop=True", 
               cv::CAP_GSTREAMER ),
    m_RecorderConsecutiveErrorCount(0),
    m_Detector(),
    m_DetectorSetting( setting ),
    m_DetectState( FaceDetector::IDLE ),
    m_PrevDetectState( FaceDetector::IDLE ),
    m_Faces(),
    m_NoDetectFaceTime(0),
    m_DetectedFaceRecorder(),
    m_WebStreamWriter()
{
    if( !m_Capture.isOpened() ){
        m_CameraState = SurveillanceCamera::ERROR_OPEN_RECORDER;
        return;
    }

    try {

        //m_Capture.set( cv::CAP_PROP_FRAME_WIDTH, 1920 );
        //m_Capture.set( cv::CAP_PROP_FRAME_HEIGHT, 1080 );
        //m_Capture.set( cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G') );
        //m_Capture.set( cv::CAP_PROP_FPS, 30 );
        if( m_DetectorSetting.Width == 0 ){
            //m_DetectorSetting.Width = static_cast<int>(1920);
            m_DetectorSetting.Width = m_Capture.get(cv::CAP_PROP_FRAME_WIDTH);
        }
        if( m_DetectorSetting.Height == 0 ){
            //m_DetectorSetting.Height = static_cast<int>(1080);
            m_DetectorSetting.Height = m_Capture.get(cv::CAP_PROP_FRAME_HEIGHT);
        }
        if( !m_Detector.Open(m_DetectorSetting) ){
            m_CameraState = SurveillanceCamera::ERROR_OPEN_RECORDER;
            return;
        }

#if 1
        auto writer = cv::VideoWriter(
            // Gstreamer output setting
            "appsrc ! autovideoconvert ! videoscale ! video/x-raw,format=I420,width=1280,height=720,framerate=30/1 ! jpegenc ! rtpjpegpay ! udpsink host=127.0.0.1 port=50001",
            cv::CAP_GSTREAMER,
            0,
            m_Capture.get( cv::CAP_PROP_FPS ),
            { static_cast<int>(m_Capture.get(cv::CAP_PROP_FRAME_WIDTH)), static_cast<int>(m_Capture.get(cv::CAP_PROP_FRAME_HEIGHT)) }
        );
        if( !writer.isOpened() ){
            m_CameraState = SurveillanceCamera::ERROR_OPEN_RECORDER;
            return;
        }
        m_WebStreamWriter = std::make_shared<ImageWriter>( writer );
        m_WebStreamWriter->Start();
#endif
    }
    catch( cv::Exception& e ){
        std::cerr << e.what() << std::endl;
        m_CameraState = SurveillanceCamera::ERROR_OPEN_RECORDER;
    }
    catch( ... ){
        m_CameraState = SurveillanceCamera::ERROR_OPEN_RECORDER;
    }
}

SurveillanceCamera::~SurveillanceCamera()
{
    m_Detector.WaitDetectResult();

    if( m_DetectedFaceRecorder.get() ){
        m_DetectedFaceRecorder->End();
    }
    if( m_WebStreamWriter.get() ){
        m_WebStreamWriter->End();
    }
}

void SurveillanceCamera::Update()
{
    switch( m_CameraState ){
    case INITIALIZING:
        ChangeSeqInitializing();
        break;
    case STREAMING:
        DoStreaming();
        ChangeSeqStreaming();
        break;
    case STREAMING_AND_RECORDING_FACES:
        DoStreamingAndRecordingFaces();
        ChangeSeqStreamingAndRecordingFaces();
        break;
    case ERROR_OPEN_RECORDER:
    case ERROR_RECORDER:
        // do nothing
        break;
    default:
        break;
    }
}

SurveillanceCamera::State SurveillanceCamera::GetState()
{
    return m_CameraState;
}

void SurveillanceCamera::ChangeSeqInitializing()
{
    // 次に進める
    m_CameraState = STREAMING;
}

void SurveillanceCamera::DoStreaming()
{
    FaceDetector::State state = FaceDetector::ERROR_FAIL_START;
    std::cout << "Streaming." << std::endl;

    try {
        cv::Mat frame;
        m_Capture >> frame;

    	std::cout << "size[]: " << frame.size().width << "," << frame.size().height << std::endl;
        m_WebStreamWriter->Enqueue( frame.clone() );

        state = DetectFace( frame );
        m_RecorderConsecutiveErrorCount = 0;
    }
    catch( cv::Exception& e ){
        std::cerr << e.what() << std::endl;
        ++m_RecorderConsecutiveErrorCount;
    }
    catch( ... ){
        ++m_RecorderConsecutiveErrorCount;
        // ログ記録
    }

    m_PrevDetectState = m_DetectState;
    m_DetectState = state;

    PrintDetectState( state );
}

FaceDetector::State SurveillanceCamera::DetectFace( cv::Mat frame )
{
    FaceDetector::State state = m_Detector.DetectResult();
    bool need_new_detect = false;

    if( state == FaceDetector::IDLE ){
        // 何もしない
    }
    else if( state == FaceDetector::OPENED ){
        need_new_detect = true;
        std::cout << "Face detect start." << std::endl;
    }
    else if(( state == FaceDetector::FACE_DETECT_NO_FACE ) ||
            ( state == FaceDetector::FACE_DETECT_OK ))
    {
        need_new_detect = true;
        m_Faces = m_Detector.GetFaceDetectVisualizedImage();
        std::cout << "Face Detected or no face." << std::endl;
    }
    else if( state == FaceDetector::FACE_DETECTING )
    {
        std::cout << "Face Detecting." << std::endl;
    }
    else if( state == FaceDetector::ERROR_FAIL_INITIALIZE )
    {
        // 顔検出モジュールの初期化に失敗しているのでどうしようもない
    }
    else if(( state == FaceDetector::ERROR_FAIL_START ) ||
            ( state == FaceDetector::ERROR_DETECT_THREAD ))
    {
        // 顔検出中のエラー
        // エラー処理が必要ならここに追加
        need_new_detect = true;
        std::cout << "Error occurred while detecting faces." << std::endl;
    }
    if( need_new_detect ){
        std::cout << "Invoke Next Detect" << std::endl;
        m_Detector.Detect( frame );
    }

    return state;
}

bool SurveillanceCamera::CreateDetectedFaceRecorder()
{
    try {
        std::string timestamp = BuildTimeStampString() + ".mp4";
        auto writer = cv::VideoWriter(
            timestamp,
            cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
            m_Capture.get(cv::CAP_PROP_FPS),
            { static_cast<int>(m_Capture.get(cv::CAP_PROP_FRAME_WIDTH)), static_cast<int>(m_Capture.get(cv::CAP_PROP_FRAME_HEIGHT)) }
        );

        if( !writer.isOpened() ){
            return false;
        }
        m_DetectedFaceRecorder = std::make_shared<ImageWriter>( writer );
        m_DetectedFaceRecorder->Start();
        if( m_DetectedFaceRecorder->IsError() ){
            return false;
        }
    }
    catch( cv::Exception& e ){
        std::cerr << e.what() << std::endl;
        return false;
    }
    catch( ... ){
        return false;
    }

    return true;
}

void SurveillanceCamera::ChangeSeqStreaming()
{
    if( m_DetectState == FaceDetector::FACE_DETECT_OK )
    {
        if( CreateDetectedFaceRecorder() ){
            m_CameraState = STREAMING_AND_RECORDING_FACES;
        }
        else {
            m_CameraState = STREAMING;
        }
    }
    else {
        m_CameraState = STREAMING;
    }

    if( IsError() ){
        m_CameraState = ERROR_RECORDER;
    }
}

void SurveillanceCamera::DoStreamingAndRecordingFaces()
{
    FaceDetector::State state = FaceDetector::ERROR_FAIL_START;
    std::cout << "Streaming And Recoding." << std::endl;

    try {
        cv::Mat frame;

        cv::TickMeter meter;
        std::cout << "Capture before" << std::endl;
        meter.start();
        m_Capture >> frame;
        meter.stop();
        std::cout << "Captured " << meter.getTimeMilli() << "[ms]" << std::endl;

        m_WebStreamWriter->Enqueue( frame.clone() );
        m_DetectedFaceRecorder->Enqueue( frame.clone() );

        state = DetectFace( frame );
        m_RecorderConsecutiveErrorCount = 0;
    }
    catch( ... ){
        ++m_RecorderConsecutiveErrorCount;
        std::cerr << "DoStreamingAndRecordingFaces() error occoured!" << std::endl;
    }

    m_PrevDetectState = m_DetectState;
    m_DetectState = state;

    if( m_DetectState == FaceDetector::FACE_DETECT_NO_FACE ){
        m_NoDetectFaceTime = std::min<uint32_t>( m_NoDetectFaceTime + 1, sk_NoDetectFaceThreshold );
    }
    else if( m_DetectState == FaceDetector::FACE_DETECT_OK ){
        m_NoDetectFaceTime = 0;
    }
}

void SurveillanceCamera::ChangeSeqStreamingAndRecordingFaces()
{
    if( m_DetectState == FaceDetector::FACE_DETECT_NO_FACE )
    {
        if( m_NoDetectFaceTime >= sk_NoDetectFaceThreshold ){
            m_NoDetectFaceTime = 0;
            if( m_DetectedFaceRecorder.get() ){
                m_DetectedFaceRecorder->End();
            }
            m_CameraState = STREAMING;
        }
    }
    else if(( m_DetectState == FaceDetector::FACE_DETECTING ) ||
            ( m_DetectState == FaceDetector::FACE_DETECT_OK ))
    {
        // 何もしない。現状維持
    }
    else {
        // エラー・もしくは想定しないステートなので録画終了
        m_NoDetectFaceTime = 0;
        if( m_DetectedFaceRecorder.get() ){
            m_DetectedFaceRecorder->End();
        }
        m_CameraState = STREAMING;
    }

    if( IsError() ){
        if( m_DetectedFaceRecorder.get() ){
            m_DetectedFaceRecorder->End();
        }
        m_CameraState = ERROR_RECORDER;
    }
}

bool SurveillanceCamera::IsError() const
{
    return  m_WebStreamWriter->IsError() || 
            (m_RecorderConsecutiveErrorCount >= sk_RecorderConsecutiveErrorThreshold);
}
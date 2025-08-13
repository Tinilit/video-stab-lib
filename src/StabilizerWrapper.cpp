#include "StabilizerWrapper.h"
#include "MotionEstimationBuilder.h"

#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>     // estimateAffinePartial2D
#include <opencv2/video/tracking.hpp> // calcOpticalFlowPyrLK
#include <opencv2/videostab.hpp>
#include <fstream>

using namespace cv;
using namespace cv::videostab;

StabilizerWrapper::StabilizerWrapper()
{
    Params params;
    params.model = "affine";
    params.local_outlier_rejection = "yes";

    MotionEstimatorL1Builder builder(params, false);
    auto motionEstimator = builder.build();

    frameSource = Ptr<FrameSourceFromQueue>(new FrameSourceFromQueue(originalFrames, mutex, condVar));
    stabilizer = makePtr<OnePassStabilizer>();  
    stabilizer->setMotionEstimator(motionEstimator);
    stabilizer->setFrameSource(frameSource);

    processingThread = std::thread([this]() {
        while (!stopFlag) {
            cv::Mat stabilized = stabilizer->nextFrame();
            if (!stabilized.empty()) {
                std::lock_guard<std::mutex> lock(mutex);
                latestStabilizedFrame = stabilized.clone();
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });
}

StabilizerWrapper::~StabilizerWrapper()
{
    stopFlag = true;
    if (processingThread.joinable()) {
        processingThread.join();
    }
}

void StabilizerWrapper::feedFrame(const cv::Mat& frame)
{
    {
        std::lock_guard<std::mutex> lock(mutex);

        const size_t MAX_QUEUE_SIZE = 10;
        if (originalFrames.size() >= MAX_QUEUE_SIZE) {
            originalFrames.pop_front();
        }

        originalFrames.push_back(frame.clone());
    }
    
    condVar.notify_one();
}

bool StabilizerWrapper::getFrame(int index, cv::Mat& out)
{
    std::lock_guard<std::mutex> lock(mutex);

    if (latestStabilizedFrame.empty())
        return false;

    out = latestStabilizedFrame.clone();
    return true;
}
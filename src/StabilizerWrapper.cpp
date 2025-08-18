#include "StabilizerWrapper.h"
#include "MotionEstimationBuilder.h"

#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>     // estimateAffinePartial2D
#include <opencv2/video/tracking.hpp> // calcOpticalFlowPyrLK
#include <opencv2/videostab.hpp>
#include "Logger.h"

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
    

    Logger::logToFile("StabilizerWrapper створено, потік обробки стартує");

    processingThread = std::thread([this]() {
        Logger::logToFile("ProcessingThread запущено");
        size_t stabilizedCount = 0;

        while (!stopFlag) {
            cv::Mat stabilized = stabilizer->nextFrame();

            if (!stabilized.empty()) {
                    std::lock_guard<std::mutex> lock(mutex);
                    Logger::logToFile("ProcessingThread: stabilizer->nextFrame() видав кадр. Розмір: " +
                      std::to_string(stabilized.rows) + "x" +
                      std::to_string(stabilized.cols));
                {

                    latestStabilizedFrame = stabilized.clone();
                    stabilizedCount++;
                }
                condVar.notify_all(); // повідомляємо getFrame, що кадр готовий

                Logger::logToFile("Стабілізований кадр оновлено. Розмір: " +
                    std::to_string(stabilized.rows) + "x" +
                    std::to_string(stabilized.cols) +
                    ", стабілізованих кадрів всього: " + std::to_string(stabilizedCount));
            } else {
                // якщо stabilizer ще не видав кадр
                Logger::logToFile("ProcessingThread: stabilizer->nextFrame() повернув порожній кадр");
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }

        Logger::logToFile("ProcessingThread завершується. Всього стабілізованих кадрів: " +
            std::to_string(stabilizedCount));
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
            Logger::logToFile("feedFrame: черга перевищила MAX_QUEUE_SIZE, видалено найстаріший кадр");
        }

        originalFrames.push_back(frame.clone());
        Logger::logToFile("feedFrame: додано кадр. Розмір черги після push: " + std::to_string(originalFrames.size()));
    }

    condVar.notify_one();
}

bool StabilizerWrapper::getFrame(int index, cv::Mat& out)
{
    std::unique_lock<std::mutex> lock(mutex);

    // чекаємо, поки з’явиться стабілізований кадр
    condVar.wait(lock, [this]() { return !latestStabilizedFrame.empty() || stopFlag; });

    if (latestStabilizedFrame.empty()) {
        Logger::logToFile("getFrame: стабілізованого кадру немає, повертаємо false");
        return false;
    }

    out = latestStabilizedFrame.clone();
    Logger::logToFile("getFrame: повертаємо стабілізований кадр. Розмір: " +
                      std::to_string(out.rows) + "x" +
                      std::to_string(out.cols) +
                      ", черга оригінальних кадрів: " + std::to_string(originalFrames.size()));
    return true;
}
#include "StabilizerWrapper.h"
#include "MotionEstimatorRansacL2Builder.h"

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

    MotionEstimatorRansacL2Builder builder(params, false);
    auto motionEstimator = builder.build();

    frameSource = Ptr<FrameSourceFromQueue>(new FrameSourceFromQueue(originalFrames, mutex, condVar));
    stabilizer = makePtr<OnePassStabilizer>();  
    stabilizer->setMotionEstimator(motionEstimator);
    stabilizer->setFrameSource(frameSource);

    stabilizer->setMotionFilter(makePtr<GaussianMotionFilter>(50, 80.0f));
    stabilizer->setTrimRatio(0.0f);
    stabilizer->setRadius(15);
    stabilizer->setBorderMode(cv::BORDER_CONSTANT);

    Logger::logToFile("StabilizerWrapper створено, потік обробки стартує");

    processingThread = std::thread([this]() {
        Logger::logToFile("ProcessingThread запущено");
        size_t stabilizedCount = 0;

        while (true) {
            cv::Mat stabilized;
            try
            {
                stabilized = stabilizer->nextFrame();

                cv::Mat diff;
                cv::absdiff(originalFrames.back(), stabilized, diff);
                double maxDiff;
                cv::minMaxLoc(diff, nullptr, &maxDiff);
                Logger::logToFile("Max difference between last original and stabilized: " + std::to_string(maxDiff));

                if (stabilized.empty())
                {
                    Logger::logToFile("ProcessingThread: stabilized frame is EMPTY");
                }
                else
                {
                    Logger::logToFile("ProcessingThread: got stabilized frame. Size = " +
                                    std::to_string(stabilized.cols) + "x" +
                                    std::to_string(stabilized.rows) +
                                    ", channels = " + std::to_string(stabilized.channels()) +
                                    ", type = " + std::to_string(stabilized.type()));
                }
            }
            catch (const cv::Exception& ex)
            {
                Logger::logToFile("ProcessingThread: cv::Exception caught in nextFrame(): " + std::string(ex.what()));
                // при необхідності можна задати stabilized = cv::Mat(); щоб не обривати потік
                stabilized = cv::Mat();
            }
            catch (const std::exception& ex)
            {
                Logger::logToFile("ProcessingThread: std::exception caught in nextFrame(): " + std::string(ex.what()));
                stabilized = cv::Mat();
            }
            catch (...)
            {
                Logger::logToFile("ProcessingThread: unknown exception caught in nextFrame()");
                stabilized = cv::Mat();
            }

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
    Logger::logToFile("getFrame: Зайшли в метод і чекаємо на стабілізований кадр.");
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
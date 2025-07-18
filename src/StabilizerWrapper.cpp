#include "StabilizerWrapper.h"

#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>     // estimateAffinePartial2D
#include <opencv2/video/tracking.hpp> // calcOpticalFlowPyrLK
#include <opencv2/videostab.hpp>
#include <fstream>

using namespace cv;
using namespace cv::videostab;

StabilizerWrapper::StabilizerWrapper()
    : processed(false)
{
}

StabilizerWrapper::~StabilizerWrapper() {}

void StabilizerWrapper::feedFrame(const cv::Mat& frame)
{
    originalFrames.push_back(frame.clone());
    processed = false;
}

bool StabilizerWrapper::processFrame()
{
    std::lock_guard<std::mutex> lock(mutex); // поточна блокування (потрібно додати mutex у клас)
    std::ofstream log("stab_log.txt", std::ios::app);

    int currentIndex = (int)originalFrames.size() - 1;
    if (currentIndex < 1)
        return false; // Потрібно мінімум 2 кадри для стабілізації

    if (currentIndex <= lastProcessedIndex)
        return false; 

    cv::Mat currFrame = originalFrames[currentIndex];
    if (currFrame.empty())
        return false;

    cv::Mat currGray;
    cvtColor(currFrame, currGray, COLOR_BGR2GRAY);

    if (!firstFrameProcessed)
    {
        prevGray = currGray.clone();
        stabilizedFrames.push_back(currFrame.clone());
        firstFrameProcessed = true;
        lastProcessedIndex = 0;
    }

    // 1. Знаходимо ключові точки та оцінюємо трансформацію
    std::vector<Point2f> prevPts, currPts;
    goodFeaturesToTrack(prevGray, prevPts, 400, 0.005, 20);

    std::vector<uchar> status;
    std::vector<float> err;
    calcOpticalFlowPyrLK(prevGray, currGray, prevPts, currPts, status, err);

    std::vector<Point2f> prevInliers, currInliers;
    for (size_t j = 0; j < status.size(); ++j)
    {
        if (status[j])
        {
            prevInliers.push_back(prevPts[j]);
            currInliers.push_back(currPts[j]);
        }
    }

    Mat transform;
    if (prevInliers.size() >= 3 && currInliers.size() >= 3)
        transform = estimateAffinePartial2D(prevInliers, currInliers);
    else
        transform = Mat::eye(2, 3, CV_64F);

    if (transform.empty())
        transform = Mat::eye(2, 3, CV_64F);

    double dx_i = transform.at<double>(0, 2);
    double dy_i = transform.at<double>(1, 2);
    double da_i = atan2(transform.at<double>(1, 0), transform.at<double>(0, 0));

    if (transform.empty())
        log << "Transform is empty! Using identity.\n";
    else
        log << "Transform:\n" << transform << std::endl;
    log << "dx=" << dx_i << ", dy=" << dy_i << ", da=" << da_i << std::endl;

    dx.push_back(dx_i);
    dy.push_back(dy_i);
    da.push_back(da_i);

    // 2. Оновлюємо траєкторію
    double prevX = (trajectoryX.empty() ? 0 : trajectoryX.back());
    double prevY = (trajectoryY.empty() ? 0 : trajectoryY.back());
    double prevA = (trajectoryA.empty() ? 0 : trajectoryA.back());

    double currX = prevX + dx_i;
    double currY = prevY + dy_i;
    double currA = prevA + da_i;

    trajectoryX.push_back(currX);
    trajectoryY.push_back(currY);
    trajectoryA.push_back(currA);

    // 3. Згладжуємо траєкторію (ковзне середнє)
    auto smoothAt = [&](int i) -> std::tuple<double, double, double> {
        double sumX = 0, sumY = 0, sumA = 0;
        int count = 0;
        for (int j = i - SMOOTH_RADIUS; j <= i + SMOOTH_RADIUS; ++j)
        {
            if (j >= 0 && j < (int)trajectoryX.size())
            {
                sumX += trajectoryX[j];
                sumY += trajectoryY[j];
                sumA += trajectoryA[j];
                count++;
            }
        }
        return {sumX / count, sumY / count, sumA / count};
    };

    smoothedX.clear();
    smoothedY.clear();
    smoothedA.clear();

    for (int i = 0; i < (int)trajectoryX.size(); ++i)
    {
        std::tuple<double, double, double> t = smoothAt(i);
        double sx = std::get<0>(t);
        double sy = std::get<1>(t);
        double sa = std::get<2>(t);
        smoothedX.push_back(sx);
        smoothedY.push_back(sy);
        smoothedA.push_back(sa);
    }

    // 4. Обчислюємо компенсуючі трансформації для останнього кадру
    int i = (int)trajectoryX.size() - 1;
    double diffX = smoothedX[i] - trajectoryX[i];
    double diffY = smoothedY[i] - trajectoryY[i];
    double diffA = smoothedA[i] - trajectoryA[i];

    double dx_comp = dx[i] + diffX;
    double dy_comp = dy[i] + diffY;
    double da_comp = da[i] + diffA;

    Mat T = Mat::eye(2, 3, CV_64F);
    T.at<double>(0, 0) = cos(da_comp);
    T.at<double>(0, 1) = -sin(da_comp);
    T.at<double>(1, 0) = sin(da_comp);
    T.at<double>(1, 1) = cos(da_comp);
    T.at<double>(0, 2) = dx_comp;
    T.at<double>(1, 2) = dy_comp;

    Mat stabilized;
    warpAffine(currFrame, stabilized, T, currFrame.size());

    stabilizedFrames.push_back(stabilized);

    prevGray = currGray.clone();
    lastProcessedIndex = currentIndex;
    processed = true;

    return processed;
}


bool StabilizerWrapper::getFrame(int index, cv::Mat& out)
{
    if (index < 0 || index >= originalFrames.size())
        return false;
    // Обробляємо кадри поки немає стабілізованого кадра з потрібним індексом
    while (index >= stabilizedFrames.size())
    {
        if (!processFrame())  // повертає false, якщо нема більше кадрів для обробки
            break;
    }

    if (index >= stabilizedFrames.size())
        return false;

    out = stabilizedFrames[index].clone();
    return true;
}
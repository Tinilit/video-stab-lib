#include "StabilizerWrapper.h"

#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>     // estimateAffinePartial2D
#include <opencv2/video/tracking.hpp> // calcOpticalFlowPyrLK
#include <opencv2/videostab.hpp>

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

void StabilizerWrapper::process()
{
    if (originalFrames.size() < 2)
        return;

    stabilizedFrames.clear();

    std::vector<double> dx, dy, da;
    std::vector<double> trajectoryX, trajectoryY, trajectoryA;

    std::vector<double> smoothedX, smoothedY, smoothedA;

    Mat prevGray;
    cvtColor(originalFrames[0], prevGray, COLOR_BGR2GRAY);

    stabilizedFrames.push_back(originalFrames[0]);

    double prevX = 0, prevY = 0, prevA = 0;

    // === 1. Обчислюємо трансформації між кадрами ===
    for (size_t i = 1; i < originalFrames.size(); ++i)
    {
        if (originalFrames[i].empty())
        {
            stabilizedFrames.push_back(originalFrames[i]);
            continue;
        }

        Mat currGray;
        cvtColor(originalFrames[i], currGray, COLOR_BGR2GRAY);

        std::vector<Point2f> prevPts, currPts;
        goodFeaturesToTrack(prevGray, prevPts, 200, 0.01, 30);

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

        dx.push_back(dx_i);
        dy.push_back(dy_i);
        da.push_back(da_i);

        prevX += dx_i;
        prevY += dy_i;
        prevA += da_i;

        trajectoryX.push_back(prevX);
        trajectoryY.push_back(prevY);
        trajectoryA.push_back(prevA);

        prevGray = currGray.clone();
    }

    // === 2. Згладжуємо траєкторію через ковзаюче середнє ===
    const int SMOOTH_RADIUS = 15;
    for (size_t i = 0; i < trajectoryX.size(); ++i)
    {
        double sumX = 0, sumY = 0, sumA = 0;
        int count = 0;
        for (int j = -SMOOTH_RADIUS; j <= SMOOTH_RADIUS; ++j)
        {
            int idx = i + j;
            if (idx >= 0 && idx < trajectoryX.size())
            {
                sumX += trajectoryX[idx];
                sumY += trajectoryY[idx];
                sumA += trajectoryA[idx];
                count++;
            }
        }

        smoothedX.push_back(sumX / count);
        smoothedY.push_back(sumY / count);
        smoothedA.push_back(sumA / count);
    }

    // === 3. Обчислюємо компенсуючі трансформації ===
    prevX = 0;
    prevY = 0;
    prevA = 0;

    for (size_t i = 1; i < originalFrames.size(); ++i)
    {
        double diffX = smoothedX[i - 1] - trajectoryX[i - 1];
        double diffY = smoothedY[i - 1] - trajectoryY[i - 1];
        double diffA = smoothedA[i - 1] - trajectoryA[i - 1];

        double dx_i = dx[i - 1] + diffX;
        double dy_i = dy[i - 1] + diffY;
        double da_i = da[i - 1] + diffA;

        Mat T = Mat::eye(2, 3, CV_64F);
        T.at<double>(0, 0) = cos(da_i);
        T.at<double>(0, 1) = -sin(da_i);
        T.at<double>(1, 0) = sin(da_i);
        T.at<double>(1, 1) = cos(da_i);
        T.at<double>(0, 2) = dx_i;
        T.at<double>(1, 2) = dy_i;

        Mat stabilized;
        warpAffine(originalFrames[i], stabilized, T, originalFrames[i].size());

        stabilizedFrames.push_back(stabilized);
    }

    processed = true;
}


bool StabilizerWrapper::getFrame(int index, cv::Mat& out)
{
    if (!processed || index < 0 || index >= stabilizedFrames.size())
        return false;

    out = stabilizedFrames[index].clone();
    return true;
}
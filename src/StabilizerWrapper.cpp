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
    processed = false; // Якщо додаємо нові кадри, треба переробити
}

void StabilizerWrapper::process()
{
    if (originalFrames.size() < 2)
        return;

    stabilizedFrames.clear();

    stabilizedFrames.push_back(originalFrames[0]);

    Mat cumulativeTransform = Mat::eye(3, 3, CV_64F);

    for (size_t i = 1; i < originalFrames.size(); ++i)
    {
        Mat prevGray, currGray;
        cvtColor(originalFrames[i - 1], prevGray, COLOR_BGR2GRAY);
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

        Mat transform = estimateAffinePartial2D(prevInliers, currInliers);

        if (transform.empty())
            transform = Mat::eye(2, 3, CV_64F);

        Mat T = Mat::eye(3, 3, CV_64F);
        transform.copyTo(T(Rect(0, 0, 3, 2)));

        cumulativeTransform = cumulativeTransform * T;

        Mat stabilized;
        warpAffine(originalFrames[i], stabilized, cumulativeTransform(Rect(0, 0, 3, 2)), originalFrames[i].size());

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
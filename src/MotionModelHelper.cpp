#include "MotionModelHelper.h"

MotionModel motionModel(const std::string &str)
{
    if (str == "transl") return MM_TRANSLATION;
    if (str == "transl_and_scale") return MM_TRANSLATION_AND_SCALE;
    if (str == "rigid") return MM_RIGID;
    if (str == "similarity") return MM_SIMILARITY;
    if (str == "affine") return MM_AFFINE;
    if (str == "homography") return MM_HOMOGRAPHY;
    throw std::runtime_error("unknown motion model: " + str);
}
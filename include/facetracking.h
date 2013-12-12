#ifndef FACETRACKING_H
#define FACETRACKING_H

#include <vector>
#include <string>
#include <opencv2/core/core.hpp>

#include "detection.h"
#include "recognition.h"
#include "human.h"

static const unsigned int FRAMES_BETWEEN_DETECTION = 50;

class FaceTracking {

public:
    FaceTracking();
    std::vector<Face> track(const cv::Mat inputImage, cv::Mat debugImage = cv::Mat());
private:
    int frameCount;

    FaceDetector facedetector;
    Recognizer faceRecognizer;
    std::vector<Human> humans;
};

#endif // FACETRACKING_H

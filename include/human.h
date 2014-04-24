#ifndef HUMAN_H
#define HUMAN_H

#include <vector>
#include <string>
#include <opencv2/core/core.hpp>

#include "detection.h"
#include "recognition.h"

enum Mode {LOST, TRACKING};

class Face;

class Human {

    friend class Face;

public:
    /** Initialize a new human, based on the bounding box of its face in the
     * current frame.
     */
    Human(const std::string& name, 
          const cv::Mat inputImage, 
          const cv::Rect boundingbox,
          Recognizer& faceRecognizer);

    /** Returns true if the given rectangle match my current bounding box.
     *
     * This is actually computed as a non-zero intersection area.
     */
    bool isMyself(const cv::Rect face) const;

    /** Returns the current analysis mode for this human: tracking, learning,...
     */
    Mode mode() const {return _mode;}

    /** Set the face bounding box, and initialize accordingly the offset between the
     * centroid of the tracked features and the boundingbox.
     */
    void relocalizeFace(const cv::Mat& image, const cv::Rect face);

    /** Update this face.
     *
     * This may mean:
     *  - track the face (ie, find the key points of the face in the current frame)
     */
    void update(const cv::Mat inputImage);

    void showFace(cv::Mat& ouputImage);

    std::string name() const {return _name;}
    cv::Matx44d pose() const {return _pose;}

private:

    std::string _name;

    // the estimate of the 6D transformation of the human head from the
    // camera perspective
    cv::Matx44d _pose;
    cv::Rect boundingbox;

    Mode _mode;

    FaceTracker tracker;
    //offset between the centroid of the tracked features and the actual face boundingbox.
    cv::Point trackerOffset;
    std::vector<cv::Point2f> features;
    Recognizer& faceRecognizer;
    bool recognizerTrained;
    
};

/** Public interface to the detected faces
 */
class Face {

public:
    Face(const Human& human):human(human) {}
    std::string name() const {return human._name;}
    cv::Rect boundingbox() const {return human.boundingbox;}
    //cv::Point center() const {return human.boundingbox.tl() + (human.boundingbox.tl() - human.boundingbox.br())/2;}
    cv::Matx44d pose() const {return human._pose;}

private:
    const Human& human;
};

#endif // HUMAN_H

#include <iostream>
#include "human.h"

#include "detection.h" // FEATURES_THRESHOLD

//#define DEBUG_human

using namespace std;
using namespace cv;

Human::Human(const string& name, 
             const Mat inputImage, 
             const Rect boundingbox,
             Recognizer& faceRecognizer) :
            _name(name),
            boundingbox(boundingbox),
            tracker(FaceTracker(inputImage,
                                FaceTracker::features(inputImage, boundingbox))),
            faceRecognizer(faceRecognizer)
{
    _mode = TRACKING;
    recognizerTrained = false;
    
    // explicit conversion from Point2f to Point
    Point centroid = tracker.centroid();
    trackerOffset = boundingbox.tl() - centroid;

}

bool Human::isMyself(const Rect face) const
{
    return (face & boundingbox).area() != 0;
}

void Human::relocalizeFace(const Mat& image, const Rect face)
{
    boundingbox = face;

    //if (_mode == LOST)
    //{
        tracker.resetFeatures(image, face);
    //}

    // explicit conversion from Point2f to Point
    Point centroid = tracker.centroid();

    trackerOffset = face.tl() - centroid;

    _mode = TRACKING;
}

void Human::update(const Mat inputImage)
{
    features = tracker.track(inputImage);

    if (features.size() < FEATURES_THRESHOLD) {
#ifdef DEBUG_human
        cout << "Not enough features! Going back to detection" << endl;
#endif
        _mode = LOST;
        return;
    }
    
    Point centroid = tracker.centroid();
    boundingbox = Rect(centroid + trackerOffset, boundingbox.size());
    boundingbox.x = max(0, boundingbox.x);
    boundingbox.y = max(0, boundingbox.y);
    boundingbox.width = min(inputImage.cols - boundingbox.x, boundingbox.width);
    boundingbox.height = min(inputImage.rows - boundingbox.y, boundingbox.height);


#ifdef DEBUG_human
    cout << "Tracking " << features.size() << " features" << endl;
#endif

    if (!recognizerTrained)
    {
        recognizerTrained = faceRecognizer.addPictureOf(inputImage(boundingbox), _name);
 
#ifdef DEBUG_human
        if (recognizerTrained) {
            cout << "Model trained for " << _name << ". I can now recognize him!" << endl;
        }
#endif
   }
}

void Human::showFace(Mat& outputImage) {

    if (_mode != LOST) {

        auto centroid = tracker.centroid();
        line( outputImage, centroid, centroid, cv::Scalar(10, 100, 200), 20 );

        for ( auto p : features ) {
            line( outputImage, p, p, cv::Scalar(10, 200, 100), 10 );
        }

        putText(outputImage,
                _name,
                centroid + Point2f(10,10),
                FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(10,100,200));
    }
    else //LOST!
    {
        putText(outputImage,
                _name + " LOST!",
                boundingbox.tl() + Point(20,20),
                FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200,100,10));
    }

    rectangle( outputImage, boundingbox, cv::Scalar(255,0,255), 4 );
}



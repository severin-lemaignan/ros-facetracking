#include <iostream>

#include "human.h"

#include "detection.h" // FEATURES_THRESHOLD
#include "face_constants.h"

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

    tracker.resetFeatures(image, face);

    // explicit conversion from Point2f to Point
    Point centroid = tracker.centroid();

    trackerOffset = face.tl() - centroid;

    _mode = TRACKING;
}

void Human::estimatePose(const Size& image_size,
                         const Point2f& leftEye, const Point2f& rightEye) {

    double focalLength = 700.;
    //Mat cameraMatrix = (cv::Mat_<double>(3,3) <<
    //    focalLength ,            0 , image_size.width /2,
    //               0 , focalLength , image_size.height/2,
    //               0,             0 , 1
    //);

    // Computation based on http://docs.opencv.org/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    // 'z' is computed first, using the 2 points (left and right pupil)
    // and resulting translation is then computed using left eye only.
    // To be more correct, the camera intrinsic matrix could be used.
    float z = focalLength * (LEFT_PUPIL.y - RIGHT_PUPIL.y) / (leftEye.y - rightEye.y);

    float tx = (z * (leftEye.x - image_size.width/2) / focalLength) - LEFT_PUPIL.x;
    float ty = (z * (leftEye.y - image_size.height/2) / focalLength) - LEFT_PUPIL.y;
    float tz = z - LEFT_PUPIL.z;


    _pose = {
        1, 0, 0, tx * .001, // matrix in meters
        0, 1, 0, ty * .001,
        0, 0, 1, tz * .001,
        0, 0, 0, 1
    };


}

void Human::update(const Mat inputImage)
{
    if (_mode == LOST) return;

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



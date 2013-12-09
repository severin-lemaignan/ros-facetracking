#include "human.h"

using namespace std;
using namespace cv;

Human::Human(const string& name, 
             const Mat inputImage, 
             const Rect boundingbox,
             Recognizer& faceRecognizer) :
            name(name),
            boundingbox(boundingbox),
            tracker(FaceTracker(inputImage,
                                FaceTracker::features(inputImage, boundingbox))),
            faceRecognizer(faceRecognizer)
{
    _mode = TRACKING;
    recognizerTrained = false;
}

bool Human::isMyself(const Rect face) const
{
    return (face & boundingbox).area() != 0;
}

void Human::update(const Mat inputImage)
{
    features = tracker.track(inputImage);
    //auto centroid = tracker.centroid();
    boundingbox = boundingRect(features);

    if (features.size() < 10) {
#ifdef DEBUG_facetracking
        cout << "Not enough features! Going back to detection" << endl;
#endif
        _mode = LOST;
        return;
    }

#ifdef DEBUG_facetracking
    cout << "Tracking " << features.size() << " features" << endl;
#endif

    if (!recognizerTrained)
    {
        recognizerTrained = faceRecognizer.addPictureOf(inputImage(boundingbox), name);
 
#ifdef DEBUG_facetracking
        if (recognizerTrained) {
            cout << "Model trained for " << name << ". I can now recognize him!" << endl;
        }
#endif
   }
}

void Human::showFace(Mat& outputImage) {

    auto centroid = tracker.centroid();
    line( outputImage, centroid, centroid, CV_RGB(10, 100, 200), 20 );

    for ( auto p : features ) {
        line( outputImage, p, p, CV_RGB(10, 200, 100), 10 );
    }

    putText(outputImage,
            name,
            centroid + Point2f(10,10),
            FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(10,100,200));


    rectangle( outputImage, boundingbox, CV_RGB(255,0,255), 4 );
}



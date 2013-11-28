#include "detection.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/video.hpp"

//#define DEBUG_detection
#ifdef DEBUG_detection
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#endif

using namespace cv;
using namespace std;

FaceDetector::FaceDetector() :
        frontalface(CascadeClassifier("haarcascade_frontalface_default.xml"))
{

#ifdef DEBUG_detection
    namedWindow("detection-debug");
#endif
}

vector<Rect> FaceDetector::detect(const Mat& image, int scaledWidth) {

    vector<Rect> faces;
    
    // Possibly shrink the image, to run much faster.
    Mat inputImg;
    float scale = image.cols / (float)scaledWidth;
    if (image.cols > scaledWidth) {
        // Shrink the image while keeping the same aspect ratio.
        int scaledHeight = cvRound(image.rows / scale);
        resize(image, inputImg, Size(scaledWidth, scaledHeight));
    }
    else {
        // Access the input image directly, since it is already small.
        inputImg = image;
    }

    equalizeHist( inputImg, inputImg );

    //-- Detect faces
    frontalface.detectMultiScale( inputImg, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

    for (auto& face : faces) {
        face.width *= scale;
        face.height *= scale;
        face.x *= scale;
        face.y *= scale;
    }
    return faces;

}

vector<Point2f> FaceDetector::features(const Mat& image, const Rect& face) {

    double quality = 0.01;
    double min_distance = 10;

    Mat mask = Mat::zeros(image.size(), CV_8UC1);
    auto rrect = RotatedRect(Point2f(face.x + face.width/2, face.y + face.height/2),
                             face.size(),
                             0.f);

    ellipse(mask, rrect, CV_RGB(255,255,255), CV_FILLED);

#ifdef DEBUG_detection
    imshow("detection-debug", mask);
#endif

    vector<Point2f> features;
    features.reserve(NB_FEATURES);

    goodFeaturesToTrack(image, features, NB_FEATURES, quality, min_distance, mask);
    return features;
}

Point2f mean(const vector<Point2f>& vals)
{
    size_t nbvals = vals.size();

    auto sum = vals[0];
    for(uint i = 1 ; i < nbvals ; ++i) sum += vals[i];
    return sum * (1.f/nbvals);
}

double variance(const vector<Point2f>& vals)
{
    size_t nbvals = vals.size();

    auto current_mean = mean(vals);

    auto temp = norm(current_mean-vals[0])*norm(current_mean-vals[0]);
    for(uint i = 1 ; i < vals.size() ; ++i)
        temp += norm(current_mean-vals[i])*norm(current_mean-vals[i]);
    return temp/nbvals;
}


FaceTracker::FaceTracker(const Mat& image,
                         const vector<Point2f>& features): 
                prevImg(image.clone()),
                prevFeatures(features),
                _centroid(mean(features)),
                _variance(variance(features))
{
#ifdef DEBUG_detection
    cout << "Initial variance of the feature cluster: " << _variance << endl;
#endif
}

vector<Point2f> FaceTracker::track(const Mat& nextImg) {

    vector<Point2f> nextFeatures;
    nextFeatures.reserve(NB_FEATURES);
    vector<unsigned char> status;
    vector<float> err;

    calcOpticalFlowPyrLK(prevImg, nextImg, 
                         prevFeatures, nextFeatures, 
                         status, err, 
                         Size(10,10), 3, 
                         TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 20, 0.01));

#ifdef DEBUG_detection
    cout << "Optical flow status: ";
    for (auto s : status) cout << (int)s << " ";
    cout << endl;
#endif


    prevImg = nextImg.clone();
    prevFeatures = nextFeatures;

    vector<Point2f> found;
    found.reserve(NB_FEATURES);

    for ( int i = 0 ; i < nextFeatures.size() ; i++ ) {
        if (status[i] == 1) found.push_back(nextFeatures[i]);
    }

    found = pruneFeatures(found);

    _centroid = mean(found);
    _variance = variance(found);
#ifdef DEBUG_detection
    cout << "Variance: " << _variance << endl;
#endif

    return found;

}


vector<Point2f> FaceTracker::pruneFeatures(const vector<Point2f>& features) {

    //auto centroid = mean(features);
    //auto variance = variance(features);

    vector<Point2f> prunedFeatures;

    for (auto p : features) {
        if (pow(norm(p - _centroid),2) < 3 * _variance) prunedFeatures.push_back(p);
    }

    return prunedFeatures;


}

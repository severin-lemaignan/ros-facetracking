#include <iostream>
#include <tuple>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>

#include "detection.h"
#include "face_constants.h"

#ifdef DEBUG
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#endif

using namespace cv;
using namespace std;

static const string face_classifier("haarcascade_frontalface_default.xml");
static const string eye_classifier("haarcascade_eye.xml");

FaceDetector::FaceDetector() :
        frontalface(CascadeClassifier(INSTALL_PREFIX + ("/share/facetracking/" + face_classifier))),
        eyes(CascadeClassifier(INSTALL_PREFIX + ("/share/facetracking/" + eye_classifier)))
{

    if (frontalface.empty()) {
        cerr << "Could not load classifier model <" << face_classifier << ">!" << endl;
        //TODO: bad in a library!!
        exit(-1);
    }

    if (eyes.empty()) {
        cerr << "Could not load classifier model <" << eye_classifier << ">!" << endl;
        //TODO: bad in a library!!
        exit(-1);
    }


#ifdef DEBUG
    namedWindow("detection-debug");
#endif
}

vector<tuple<Rect, Point, Point>> FaceDetector::detect(const Mat& image, int scaledWidth) {

    vector<Rect> rawfaces;
    vector<tuple<Rect, Point, Point>> faces;

    Point leftEye, rightEye;
    
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
    frontalface.detectMultiScale( inputImg, rawfaces, 1.1, 2, 0, Size(30, 30) );


    for (auto& face : rawfaces) {

        face.width *= scale;
        face.height *= scale;
        face.x *= scale;
        face.y *= scale;

        // only keep face if the eyes are detected as well
        if (detectBothEyes(image(face), leftEye, rightEye, true)) {
                faces.push_back(make_tuple(face, 
                                           leftEye + face.tl(), 
                                           rightEye + face.tl()));
        }

    }
    return faces;

}

bool FaceDetector::detectBothEyes(const Mat &face, 
                                  Point &leftEye, Point &rightEye, 
                                  bool relaxed) const
{
    // Skip the borders of the face, since it is usually just hair and ears, that we don't care about.
    
    // For default eye.xml or eyeglasses.xml: Finds both eyes in roughly 40% of detected faces, but does not detect closed eyes.

    float EYE_SX = 0.16f;
    float EYE_SY = 0.26f;
    float EYE_SW = 0.30f;
    float EYE_SH = 0.28f;

    // in 'relaxed' mode, we look for the eyes in a larger portion of the
    // image. Takes more time.
    if (relaxed) {
        EYE_SX = 0.1f;
        EYE_SY = 0.2f;
        EYE_SW = 0.40f;
        EYE_SH = 0.4f;
    }

    int leftX = cvRound(face.cols * EYE_SX);
    int topY = cvRound(face.rows * EYE_SY);
    int widthX = cvRound(face.cols * EYE_SW);
    int heightY = cvRound(face.rows * EYE_SH);
    int rightX = cvRound(face.cols * (1.0-EYE_SX-EYE_SW) );  // Start of right-eye corner

    Mat topLeftOfFace = face(Rect(leftX, topY, widthX, heightY));
    Mat topRightOfFace = face(Rect(rightX, topY, widthX, heightY));

    vector<Rect> leftEyeRects, rightEyeRects;
    Rect leftEyeRect, rightEyeRect;

#ifdef DEBUG
    Mat debugImage = face.clone();
    rectangle(debugImage, Rect(leftX, topY, widthX, heightY), cv::Scalar(255,255,255), 3);
    rectangle(debugImage, Rect(rightX, topY, widthX, heightY), cv::Scalar(255,255,255), 3);
#endif


    int flags = CASCADE_FIND_BIGGEST_OBJECT;
    eyes.detectMultiScale( topLeftOfFace, leftEyeRects, 1.1, 2, flags, Size(30, 30) );
    eyes.detectMultiScale( topRightOfFace, rightEyeRects, 1.1, 2, flags, Size(30, 30) );

    if (   leftEyeRects.size() == 0
        || rightEyeRects.size() == 0) // Check if the eye was detected.
    {
        return false;
    }
    leftEyeRect = leftEyeRects[0];
    rightEyeRect = rightEyeRects[0];

    leftEyeRect.x += leftX;    // Adjust the left-eye rectangle because the face border was removed.
    leftEyeRect.y += topY;
    leftEye = Point(leftEyeRect.x + leftEyeRect.width/2, leftEyeRect.y + leftEyeRect.height/2);

    rightEyeRect.x += rightX; // Adjust the right-eye rectangle, since it starts on the right side of the image.
    rightEyeRect.y += topY;  // Adjust the right-eye rectangle because the face border was removed.
    rightEye = Point(rightEyeRect.x + rightEyeRect.width/2, rightEyeRect.y + rightEyeRect.height/2);

#ifdef DEBUG
    line(debugImage, leftEye,leftEye, cv::Scalar(255,255,255), 3);
    line(debugImage, rightEye,rightEye, cv::Scalar(255,255,255), 3);
    namedWindow("eyes-debug");
    imshow("eyes-debug", debugImage);
#endif

    return true;
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
#ifdef DEBUG
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

#ifdef DEBUG
    cout << "Optical flow status: ";
    for (auto s : status) cout << (int)s << " ";
    cout << endl;
#endif


    prevImg = nextImg.clone();

    vector<Point2f> found;
    found.reserve(NB_FEATURES);

    for ( int i = 0 ; i < nextFeatures.size() ; i++ ) {
        if (status[i] == 1) found.push_back(nextFeatures[i]);
    }


    if (!found.empty()) {

        _centroid = mean(found);
        // do not recompute the variance. Keep the original value computed when the face
        // tracker is created or reset.
    }

    found = pruneFeatures(found);
    prevFeatures = found;
    return found;

}

void FaceTracker::resetFeatures(const Mat& image, const Rect& face) {
    prevImg = image.clone();
    prevFeatures = features(image, face);
    _centroid = mean(prevFeatures);
    _variance = variance(prevFeatures);
}

vector<Point2f> FaceTracker::features(const Mat& image, const Rect& face) {

    double quality = 0.1;
    double min_distance = 15;

    Mat mask = Mat(image.size(), CV_8U, Scalar(0)); // Start with an empty mask.
    Point faceCenter = Point( face.x + face.width/2, 
                              face.y + face.height * FACE_ELLIPSE_CY);
    Size size = Size( cvRound(face.size().width * FACE_ELLIPSE_W), 
                      cvRound(face.size().height * FACE_ELLIPSE_H) );
    ellipse(mask, faceCenter, size, 0, 0, 360, Scalar(255), -1); // tickness=-1 -> filled
 
    vector<Point2f> features;
    features.reserve(NB_FEATURES);

    goodFeaturesToTrack(image, features, NB_FEATURES, quality, min_distance, mask);

#ifdef DEBUG
    Mat debugImage;
    image.copyTo(debugImage, mask);
    
    cvtColor(debugImage, debugImage, cv::COLOR_GRAY2BGR);

    rectangle( debugImage, face, cv::Scalar(0,0,255), 4 );


    for ( auto p : features ) {
        line( debugImage, p, p, cv::Scalar(10, 200, 100), 10 );
    }

    imshow("detection-debug", debugImage);
#endif

    return features;
}

/**
 * Only keep features 'close enough' to the feature cloud centroid.
 * 'close' is dependent on the initial variance of the cloud.
 */
vector<Point2f> FaceTracker::pruneFeatures(const vector<Point2f>& features) {

    //auto centroid = mean(features);
    //auto variance = variance(features);

    vector<Point2f> prunedFeatures;

    for (auto p : features) {
        if (pow(norm(p - _centroid),2) < 3 * _variance) prunedFeatures.push_back(p);
    }

    return prunedFeatures;


}

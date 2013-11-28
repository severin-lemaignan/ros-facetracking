#include <vector>
#include <map>
#include <string>
#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"

// max nb of image per user we want to train the models on.
static const int MAX_TRAINING_IMAGES = 5;

static const int FACE_WIDTH = 200;

class Recognizer {

public:
    Recognizer();

    /** Add data to the training set, and train the model as soon as enough images
     * are available for a given label.
     *
     * Returns true if enough images are already collected for this label.
     */
    bool addPictureOf(const cv::Mat& image, const std::string& label);

    /** Returns a pair {label, confidence}
     */
    std::pair<std::string, double> whois(const cv::Mat& image);


private:
    // we need to detect the eyes to preprocess the face before learning
    cv::CascadeClassifier eyes;

    bool preprocessFace(const cv::Mat& inputImage, cv::Mat& outputImage);
    bool detectBothEyes(const cv::Mat &face, cv::Point &leftEye, cv::Point &rightEye);
    void train(int label);

    cv::Ptr<cv::FaceRecognizer> model;

    std::map<int, std::vector<cv::Mat>> trainingSet;
    std::map<int, bool> trained_labels;
    std::map<int, std::string> human_labels;
};


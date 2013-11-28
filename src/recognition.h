#include <vector>
#include <map>
#include <string>
#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"

// max nb of image per user we want to train the models on.
static const int MAX_TRAINING_IMAGES = 10;

class Recognizer {

    Recognizer();

public:

    /**
     * Returns false if enough images are already collected for this label.
     */
    bool addPictureOf(const cv::Mat& image, const std::string& label);

    void train();

    /** Returns a pair {label, confidence}
     */
    std::pair<std::string, double> predict(const cv::Mat& image);

private:
    bool trained = false;

    cv::Ptr<cv::FaceRecognizer> model;

    std::map<int, std::vector<cv::Mat>> trainingSet;
    std::map<int, std::string> human_labels;
};


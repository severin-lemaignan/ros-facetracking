#include <utility> // make_pair

#include "recognition.h"

using namespace cv;
using namespace std;

Recognizer::Recognizer():
                trained(false)
{

    int num_components = 10;
    double threshold = 10.0;

    model = createEigenFaceRecognizer(num_components, threshold);
}

bool Recognizer::addPictureOf(const Mat& image, const string& label) {

    int idx = human_labels.size();

    for (const auto& kv : human_labels) {
        if (label == kv.second) {
            idx = kv.first;
            break;
        }
    }


    if (trainingSet[idx].size() < MAX_TRAINING_IMAGES) {
        trainingSet[idx].push_back(image.clone());
        return true;
    }
    return false;
}

void Recognizer::train() {
    vector<Mat> images;
    vector<int> labels;
    for (const auto& kv : trainingSet) {
        for (const auto& image : kv.second) {
            labels.push_back(kv.first);
            images.push_back(image);
        }
    }

    model->train(images, labels);
    trained = true;
}

pair<string, double> Recognizer::predict(const Mat& image) {

    int label;
    double confidence;

    model->predict(image, label, confidence);

    return make_pair(human_labels[label], confidence);
}


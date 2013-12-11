#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <sstream>
#include <vector>
#include <map>

#include "detection.h"
#include "recognition.h"
#include "human.h"

#define DEBUG_facetracking

using namespace cv;
using namespace std;

static const unsigned int FRAMES_BETWEEN_DETECTION = 50;


int main(int argc, char *argv[])
{

    FaceDetector facedetector;

    cout << "Compiled with OpenCV version " << CV_VERSION << endl << endl;

    // Allow the user to specify a camera number, since not all computers will be the same camera number.
    int cameraIndex = 0;   // Change this if you want to use a different camera device.
    if (argc > 1) {
        cameraIndex = atoi(argv[1]);
    }

    // The source of input images
    cv::VideoCapture videoCapture(cameraIndex);
    if (!videoCapture.isOpened())
    {
        std::cerr << "Unable to initialise video capture." << std::endl;
        return 1;
    }

    videoCapture.set(CV_CAP_PROP_FRAME_WIDTH, 640);
    videoCapture.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

    // Print an empty lines to leave room for display with ANSI sequences
    cout << endl << endl;

    namedWindow("faces"); 

    Mat cameraImage, inputImage;

    unsigned int frameCount = 0;

    Recognizer faceRecognizer;

    vector<Human> humans;

    bool learn = true;

    // Main loop, exiting when 'q is pressed'
    for (; 'q' != (char) cv::waitKey(1); ) {

        // Capture a new image.
        videoCapture.read(cameraImage);
        auto debugImage = cameraImage.clone();

        cvtColor(cameraImage, inputImage, CV_BGR2GRAY);

        int64 tStartCount = cv::getTickCount();

        // Force detection every few second to be able to detect new users.
        if (frameCount % FRAMES_BETWEEN_DETECTION == 0) {

            auto faces = facedetector.detect(inputImage);

            for( const auto& face : faces )
            {
                bool alreadyTracked = false;
                for(auto& human : humans) {
                    if (human.isMyself(face))
                    {
                        human.relocalizeFace(inputImage, face);
                        alreadyTracked = true;
                        break;
                    }
                }
                if (alreadyTracked) continue;

                // if we come here, a new face has been detected
                // first check if we recognize it.
                // if not, create a new human
                auto guess = faceRecognizer.whois(inputImage(face));
                if (guess.second != 0.) {
                    cout << "\x1b[1F\t\t\tI think this is " << guess.first << " (confidence: " << guess.second << ")" << endl;
                    for (auto& human : humans) {
                        if (human.name() == guess.first)
                        {
                            human.relocalizeFace(inputImage, face);
                            break;
                        }
                    }
                } else {
                    cout << "\x1b[1F\t\t\tI do not recognize this face! Creating new human" << endl;

                    stringstream namestr;
                    namestr << "human" << humans.size() + 1;
                    humans.push_back(Human(namestr.str(), inputImage, face, faceRecognizer));
                }
            }

        }

        // face tracking!
        for( auto human : humans) {
            human.update(inputImage);
#ifdef DEBUG_facetracking
            human.showFace(debugImage);
#endif
        }

#ifndef DEBUG_facetracking
        cout << "\x1b[1F\x1b[1F"; // clear line above + up 2 lines
#endif
        cout << "Time to detect faces: " << ((double)cv::getTickCount() - tStartCount)/cv::getTickFrequency() << "ms" << std::endl;
        cout << humans.size() << " face(s) detected." << endl;

        // Finally...
#ifdef DEBUG_facetracking
        cv::imshow("faces", debugImage);
#endif
        frameCount++;
    }

    cv::destroyWindow("faces");
    videoCapture.release();

}

#include <opencv2/core/core.hpp>
#include <iostream>
#include <sstream>
#include <vector>

#include "facetracking.h"

using namespace cv;
using namespace std;


FaceTracking::FaceTracking() : frameCount(0), faceRecognizer(facedetector) {}

vector<Face> FaceTracking::track(const Mat inputImage, Mat debugImage)
{
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
                cout << "I think this is " << guess.first << " (confidence: " << guess.second << ")" << endl;
                for (auto& human : humans) {
                    if (human.name() == guess.first)
                    {
                        human.relocalizeFace(inputImage, face);
                        break;
                    }
                }
            } else {
                cout << "I do not recognize this face! Creating new human" << endl;

                stringstream namestr;
                namestr << "human" << humans.size() + 1;
                humans.push_back(Human(namestr.str(), inputImage, face, faceRecognizer));
            }
        }

    }

    vector<Face> faces;
    // face tracking!
    for( auto human : humans) {
        human.update(inputImage);
        faces.push_back(Face(human));
        if (!debugImage.empty()) human.showFace(debugImage);
    }

    frameCount++;

    return faces;



}


#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#ifdef OPENCV3
#include <opencv2/core/utility.hpp> // getTickCount
#endif
#include <iostream>

// include log4cxx header files.
#include "log4cxx/logger.h"
#include "log4cxx/basicconfigurator.h"


#include "facetracking.h"

using namespace cv;
using namespace std;
using namespace log4cxx;

LoggerPtr logger(Logger::getLogger("facetracking"));

const static string test1 = "single_user_static_camera.avi";

int main(int argc, char *argv[])
{

    int wait = 0; //time in millisecond to wait between frame. Default to no wait (-> nothing will be displayed).

    if (argc > 1) {
        wait = atoi(argv[1]);
    }
    // Set up a simple configuration that logs on the console.
    BasicConfigurator::configure();
    
    LOG4CXX_INFO(logger, "Compiled with OpenCV version " << CV_VERSION);

    cv::VideoCapture videoCapture(test1);
    if (!videoCapture.isOpened())
    {
        LOG4CXX_ERROR(logger, "Unable to initialise video capture.");
        return 1;
    }

    namedWindow("faces"); 

    FaceTracking facetracking;

    Mat cameraImage, inputImage;


    while(videoCapture.read(cameraImage)) {

        auto debugImage = cameraImage.clone();

        cvtColor(cameraImage, inputImage, cv::COLOR_BGR2GRAY);

        int64 tStartCount = cv::getTickCount();

        auto humans = facetracking.track(inputImage, debugImage);

        //cout << "Time to detect faces: " << ((double)cv::getTickCount() - tStartCount)/cv::getTickFrequency() * 1000. << "ms" << std::endl;
        //cout << humans.size() << " face(s) detected." << endl;

        imshow("faces", debugImage);
        if (wait > 0) waitKey(wait);
    }

    cv::destroyWindow("faces");
    videoCapture.release();

}

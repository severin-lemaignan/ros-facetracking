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

int main(int argc, char *argv[])
{

    // Set up a simple configuration that logs on the console.
    BasicConfigurator::configure();
    
    LOG4CXX_INFO(logger, "Compiled with OpenCV version " << CV_VERSION);

    // Allow the user to specify a camera number, since not all computers will be the same camera number.
    int cameraIndex = 0;   // Change this if you want to use a different camera device.
    if (argc > 1) {
        cameraIndex = atoi(argv[1]);
    }

    // The source of input images
    cv::VideoCapture videoCapture(cameraIndex);
    if (!videoCapture.isOpened())
    {
        LOG4CXX_ERROR(logger, "Unable to initialise video capture.");
        return 1;
    }

#ifdef OPENCV3
    videoCapture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    videoCapture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
#else
    videoCapture.set(CV_CAP_PROP_FRAME_WIDTH, 640);
    videoCapture.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
#endif

    namedWindow("faces"); 

    FaceTracking facetracking;

    Mat cameraImage, inputImage;


    // Main loop, exiting when 'q is pressed'
    for (; 'q' != (char) cv::waitKey(1); ) {

        // Capture a new image.
        videoCapture.read(cameraImage);
        auto debugImage = cameraImage.clone();

        cvtColor(cameraImage, inputImage, cv::COLOR_BGR2GRAY);

        int64 tStartCount = getTickCount();

        auto humans = facetracking.track(inputImage, debugImage);

        cout << "Time to detect faces: " << ((double)getTickCount() - tStartCount)/getTickFrequency() * 1000. << "ms" << std::endl;
        cout << humans.size() << " face(s) detected." << endl;

        imshow("faces", debugImage);
    }

    cv::destroyWindow("faces");
    videoCapture.release();

}

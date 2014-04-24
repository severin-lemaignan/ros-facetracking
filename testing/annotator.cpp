#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <unistd.h> //usleep

using namespace cv;
using namespace std;

Mat img, feedback;
Point prevPt(-1,-1);

struct TrackingInfo {

    TrackingInfo(string file) :
        fs(file + ".yml", FileStorage::WRITE)
    {
    }

    void start(int frame) 
    {
        fs << "START" << frame;
    }

    void stop(int frame)
    {
        fs << "STOP" << frame;
    }

private:
    FileStorage fs;
};

static void onMouse( int event, int x, int y, int flags, void* )
{
    if( event == cv::EVENT_LBUTTONUP || !(flags & cv::EVENT_FLAG_LBUTTON) )
        prevPt = Point(-1,-1);
    else if( event == cv::EVENT_LBUTTONDOWN )
        prevPt = Point(x,y);
    else if( event == cv::EVENT_MOUSEMOVE && (flags & cv::EVENT_FLAG_LBUTTON) )
    {
        Point pt(x,y);
        if( prevPt.x < 0 )
            prevPt = pt;

        feedback = img.clone();
        rectangle( feedback, prevPt, pt, Scalar::all(255), 2, 8, 0 );
    }
}


int main(int argc, char *argv[])
{

    int fps = 15;

    string file;

    if (argc > 1) {
        file = string(argv[1]);
    }
    else {
        cout << "Usage: annotator <video.avi>"<< endl;
        return 1;
    }

    TrackingInfo tracker(file);
    cout << "Compiled with OpenCV version " << CV_VERSION << endl;

    cv::VideoCapture videoCapture(file);
    if (!videoCapture.isOpened())
    {
        cerr << "Unable to initialise video capture." << endl;
        return 1;
    }

    cout << "Space to pause, up/down arrows to speed up/down, q or ESC to quit" << endl;

    namedWindow(file);
    setMouseCallback(file, onMouse, 0);


    int frame = 0;
    char key = 0;

    bool running = true;
    bool pause = false;

    tracker.start(1);

    while(running) {
        if(!pause) {
            bool more = videoCapture.read(img);
            if (!more) break;
            frame++;
            feedback = img.clone();
        }


        key = (char) waitKey(1./fps * 1000.);
        //cout << "Key: " << std::hex << (int)key << endl;
        switch(key) {
            case ' ':
                pause = !pause;
                break;
            case 'q':
            case 0x1b:
                running = false;
                break;
            case 0x52:
                fps+=5;
                break;
            case 0x54:
                fps = max(1, fps-5);
                break;
        }
        key = 0;

        imshow(file, feedback);
    }
    tracker.stop(frame);

    cv::destroyWindow(file);
    videoCapture.release();

}

#include <string>

// ROS
#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include "facetracking.h"

// how many second in the *future* the markers transformation should be published?
// this allow to compensate for the 'slowness' of tag detection, but introduce
// some lag in TF.
#define TRANSFORM_FUTURE_DATING 0


using namespace std;
using namespace cv;

class ROSFaceTracker {

    ros::NodeHandle& rosNode;
    image_transport::ImageTransport it;
    image_transport::CameraSubscriber sub;

    tf::TransformBroadcaster br;
    tf::Transform transform;
    std::string camera_frame;

    Mat inputImage;

    FaceTracking facetracking;

public:

    ROSFaceTracker(ros::NodeHandle& rosNode,
                      const string& camera_frame):
            rosNode(rosNode),
            it(rosNode),
            camera_frame(camera_frame)
    {

        sub = it.subscribeCamera("image", 1, &ROSFaceTracker::track, this);
    }

    void setROSTransform(Matx44d trans, tf::Transform& transform)
    {
        transform.setOrigin( tf::Vector3( trans(0,3) / 1000,
                                        trans(1,3) / 1000,
                                        trans(2,3) / 1000) );

        tf::Quaternion qrot;
        tf::Matrix3x3 mrot(
            trans(0,0), trans(0,1), trans(0,2),
            trans(1,0), trans(1,1), trans(1,2),
            trans(2,0), trans(2,1), trans(2,2));
        mrot.getRotation(qrot);
        transform.setRotation(qrot);
    }

    void track(const sensor_msgs::ImageConstPtr& msg, 
               const sensor_msgs::CameraInfoConstPtr& camerainfo)
    {
        // hopefully no copy here:
        //  - assignement operator of cv::Mat does not copy the data
        //  - toCvShare does no copy if the default (source) encoding is used.
        inputImage = cv_bridge::toCvShare(msg, "mono8")->image; 


        auto humans = facetracking.track(inputImage);

        ROS_DEBUG_STREAM(humans.size() << " humans found.");
        
        // do smthg intelligent here
        for (auto& human : humans) {

            setROSTransform(human.pose(), 
                            transform);


            br.sendTransform(
                    tf::StampedTransform(transform, 
                                        ros::Time::now() + ros::Duration(TRANSFORM_FUTURE_DATING), 
                                        camera_frame, 
                                        human.name()));
        }


    }


};


int main(int argc, char* argv[])
{
    //ROS initialization
    ros::init(argc, argv, "ros_facetracking");
    ros::NodeHandle rosNode;
    ros::NodeHandle _private_node("~");

    // load parameters
    string camera_frame;
    _private_node.param<string>("camera_frame_id", camera_frame, "camera");


    // initialize the detector by subscribing to the camera video stream
    ROSFaceTracker tracker(rosNode, camera_frame);
    ROS_INFO("ros_facetracking is ready. Humans locations will be published on TF when detected.");
    ros::spin();

    return 0;
}


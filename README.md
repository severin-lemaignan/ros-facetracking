ros-facetracking
================

(Yet Another) fast face tracker/face recognizer for ROS based on OpenCV. Contrary to other face detection nodes (like [face_detector](http://wiki.ros.org/face_detector)), it relies only on monocular vision. This is more prone to false positive, but does not require a RGBD camera.

It features Haar-based face detection (stock OpenCV), face tracking based on optical flow (idea borrowed from the [pi_face_tracker](http://wiki.ros.org/pi_face_tracker]) and face recognition (as demonstrated in the book [Mastering OpenCV](https://github.com/MasteringOpenCV/code/tree/master/Chapter8_FaceRecognition)).

So, nothing new, but several good algorithm in one lightweight package.

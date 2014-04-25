
// Constants describing typical faces layout
const double FACE_ELLIPSE_CY = 0.40;
const double FACE_ELLIPSE_W = 0.50;         // Should be atleast 0.5
const double FACE_ELLIPSE_H = 0.80;         // Controls how tall the face mask is.

// 3D translation of the pupils from the head frame (in millimeters)
// ~standard values taken from http://en.wikipedia.org/wiki/Human_head
// Here, the head origin is ~the center of the head
const cv::Point3f LEFT_PUPIL = cv::Point3f(0.f, 32.f, 100.f);
const cv::Point3f RIGHT_PUPIL = cv::Point3f(0.f, -32.f, 100.f);


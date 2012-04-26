#include <ecto/ecto.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>

#include <image_geometry/stereo_camera_model.h>
#include <camera_calibration_parsers/parse.h>

namespace features3d
{

using ecto::tendrils;

struct StereoDepthEstimator
{
  static void declare_params(tendrils& params)
  {
     params.declare<std::string>("camera_info_left_file", "The left camera calibration file. Typically a .yaml", "camera_left.yaml");
     params.declare<std::string>("camera_info_right_file", "The right camera calibration file. Typically a .yaml", "camera_right.yaml");
  }

  static void declare_io(const tendrils& params, tendrils& inputs, tendrils& outputs)
  {
    inputs.declare<sensor_msgs::CameraInfoConstPtr>("camera_info_left", "Left camera info.");
    inputs.declare<sensor_msgs::CameraInfoConstPtr>("camera_info_right", "Right camera info.");
    inputs.declare<std::vector<cv::Point2f> >("points_left", "Left image points.").required(true);
    inputs.declare<std::vector<cv::Point2f> >("points_right", "Right image points.").required(true);
    outputs.declare<std::vector<cv::Point3d> >("points_3d", "The triangulated 3d points.");
  }

  void configure(const tendrils& params, const tendrils& inputs, const tendrils& outputs)
  {
    std::string left_calibration_file = params.get<std::string>("camera_info_left_file");
    std::string right_calibration_file = params.get<std::string>("camera_info_right_file");

    if (left_calibration_file.length() != 0 && right_calibration_file.length() != 0)
    {
      std::cout << "loading camera info from " << left_calibration_file << " and " << right_calibration_file << std::endl;
      std::string left_camera_name, right_camera_name;
      sensor_msgs::CameraInfo info_left, info_right;
      camera_calibration_parsers::readCalibration(left_calibration_file, left_camera_name, info_left);
      camera_calibration_parsers::readCalibration(right_calibration_file, right_camera_name, info_right);
      stereo_camera_model_.fromCameraInfo(info_left, info_right);
    }
  }
 
  int process(const tendrils& inputs, const tendrils& outputs)
  {
    sensor_msgs::CameraInfoConstPtr info_left;
    inputs["camera_info_left"] >> info_left;
    sensor_msgs::CameraInfoConstPtr info_right;
    inputs["camera_info_right"] >> info_right;
    if (info_left && info_right)
    {
      std::cout << "using camera info topic" << std::endl;
      stereo_camera_model_.fromCameraInfo(*info_left, *info_right);
    }
 
    std::vector<cv::Point2f> left_points, right_points;
    inputs["points_left"] >> left_points;
    inputs["points_right"] >> right_points;
    std::vector<cv::Point3d> points_3d;
    points_3d.resize(left_points.size());
    for (size_t i = 0; i < left_points.size(); ++i)
    {
      double disparity = left_points[i].x - right_points[i].x;
      stereo_camera_model_.projectDisparityTo3d(left_points[i], disparity, points_3d[i]);
    }
    outputs["points_3d"] << points_3d;
    return ecto::OK;
  }

private:

  image_geometry::StereoCameraModel stereo_camera_model_;

};

}

ECTO_CELL(features3d, features3d::StereoDepthEstimator, "StereoDepthEstimator", "Projects matched 2d points from left and right image to 3d points.");


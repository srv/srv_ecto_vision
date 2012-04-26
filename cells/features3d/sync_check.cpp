#include <ecto/ecto.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>

#include <image_geometry/stereo_camera_model.h>
#include <sensor_msgs/Image.h>

namespace features3d
{

using ecto::tendrils;

struct SyncCheck
{
  static void declare_params(tendrils& params)
  {
  }

  static void declare_io(const tendrils& params, tendrils& inputs, tendrils& outputs)
  {
    inputs.declare<sensor_msgs::CameraInfoConstPtr>("camera_info_left", "Left camera info.").required(true);
    inputs.declare<sensor_msgs::CameraInfoConstPtr>("camera_info_right", "Right camera info.").required(true);
    inputs.declare<sensor_msgs::ImageConstPtr>("image_left", "Left image.").required(true);
    inputs.declare<sensor_msgs::ImageConstPtr>("image_right", "Right image.").required(true);
  }

  void configure(const tendrils& params, const tendrils& inputs, const tendrils& outputs)
  {
  }
 
  int process(const tendrils& inputs, const tendrils& outputs)
  {
    sensor_msgs::CameraInfoConstPtr info_left;
    inputs["camera_info_left"] >> info_left;
    sensor_msgs::CameraInfoConstPtr info_right;
    inputs["camera_info_right"] >> info_right;
 
    sensor_msgs::ImageConstPtr image_left, image_right;
    inputs["image_left"] >> image_left;
    inputs["image_right"] >> image_right;

    std::cout << image_left->header.stamp << " " << image_right->header.stamp << std::endl;

    return ecto::OK;
  }

private:

};

}

ECTO_CELL(features3d, features3d::SyncCheck, "SyncCheck", "Check inputs for sync.");


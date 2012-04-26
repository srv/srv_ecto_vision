#include <ecto_pcl/ecto_pcl.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>

namespace features3d
{

using ecto::tendrils;

struct PointsToPointCloud
{
  static void declare_params(tendrils& params)
  {
  }

  static void declare_io(const tendrils& params, tendrils& inputs, tendrils& outputs)
  {
    inputs.declare<std::vector<cv::Point3d> >("points_3d", "Input 3D points.").required(true);
    inputs.declare<cv::Mat>("image", "Input image, used for color.");
    inputs.declare<std::vector<cv::Point2f> >("image_points", "Image points, correspondig to points_3d.");
    outputs.declare<ecto::pcl::PointCloud>("point_cloud", "The (colored) output cloud.");
  }

  void configure(const tendrils& params, const tendrils& inputs, const tendrils& outputs)
  {
    output_ = outputs["point_cloud"];
  }
 
  int process(const tendrils& inputs, const tendrils& outputs)
  {
    std::vector<cv::Point2f> image_points;
    inputs["image_points"] >> image_points;
    std::vector<cv::Point3d> points_3d;
    inputs["points_3d"] >> points_3d;
    cv::Mat image;
    inputs["image"] >> image;
    pcl::PointCloud<pcl::PointXYZRGB> point_cloud;
    for (size_t i = 0; i < points_3d.size(); ++i)
    {
      pcl::PointXYZRGB point;
      point.x = points_3d[i].x;
      point.y = points_3d[i].y;
      point.z = points_3d[i].z;
      // pack r/g/b into rgb
      if (!image.empty() && image_points.size() != 0)
      {
        cv::Vec3b color = image.at<cv::Vec3b>(image_points[i].y, image_points[i].x);
        uint32_t rgb = ((uint32_t)color[0] << 16 | (uint32_t)color[1] << 8 | (uint32_t)color[2]);
        point.rgb = *reinterpret_cast<float*>(&rgb);
      }
      point_cloud.push_back(point);
    }
    *output_ = ecto::pcl::xyz_cloud_variant_t(point_cloud.makeShared());
    return ecto::OK;
  }

private:

  ecto::spore<ecto::pcl::PointCloud> output_;

};

}

ECTO_CELL(features3d, features3d::PointsToPointCloud, "PointsToPointCloud", "Convert a vector of cv points and corresponding image points to a colored pcl point cloud.");


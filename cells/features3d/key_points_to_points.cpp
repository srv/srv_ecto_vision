#include <ecto/ecto.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>

namespace features3d
{

using ecto::tendrils;

struct KeyPointsToPoints
{
  static void declare_params(tendrils& params)
  {
  }

  static void declare_io(const tendrils& params, tendrils& inputs, tendrils& outputs)
  {
    inputs.declare(&KeyPointsToPoints::in_key_points, "key_points", "The input key points.").required(true);
    inputs.declare(&KeyPointsToPoints::in_indices, "indices", "The input indices, may be empty.");
    outputs.declare(&KeyPointsToPoints::out_points, "points", "The output points.");
  }

  void configure(const tendrils& params, const tendrils& inputs, const tendrils& outputs)
  {
  }
 
  int process(const tendrils& input, const tendrils& output)
  {
    *out_points = std::vector<cv::Point2f>();
    cv::KeyPoint::convert(*in_key_points, *out_points, *in_indices);
    return ecto::OK;
  }

  ecto::spore<std::vector<cv::KeyPoint> > in_key_points;
  ecto::spore<std::vector<int> > in_indices;
  ecto::spore<std::vector<cv::Point2f> > out_points;
};

}

ECTO_CELL(features3d, features3d::KeyPointsToPoints, "KeyPointsToPoints", "Extracts points from key points.");


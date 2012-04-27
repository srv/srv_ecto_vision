#include <ecto/ecto.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>

namespace features3d
{

using ecto::tendrils;

struct Transform
{
  static void declare_io(const tendrils& params, tendrils& inputs, tendrils& outputs)
  {
    inputs.declare<cv::Mat>("mat", "Input matrix, must have as many channels (1 to 4) as transform.cols or transform.cols-1.").required(true);
    inputs.declare<cv::Mat>("transformation", "Transformation to apply.").required(true);
    outputs.declare<cv::Mat>("mat", "Transformed output.");
  }

  int process(const tendrils& inputs, const tendrils& outputs)
  {
    cv::Mat in, out, transform;
    inputs["mat"] >> in;
    inputs["transformation"] >> transform;
    cv::transform(in, out, transform);
    outputs["mat"] << out;
    return ecto::OK;
  }
};

}

ECTO_CELL(features3d, features3d::Transform, "Transform", "Applies a transform to an input matrix.");


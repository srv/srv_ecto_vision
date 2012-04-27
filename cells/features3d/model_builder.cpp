#include <ecto/ecto.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>

namespace features3d
{

using ecto::tendrils;

struct ModelBuilder
{
  static void declare_params(tendrils& params)
  {
  }

  static void declare_io(const tendrils& params, tendrils& inputs, tendrils& outputs)
  {
    inputs.declare<cv::Mat>("points", "Input source points.").required(true);
    inputs.declare<cv::Mat>("descriptors", "Input descriptors.").required(true);
    outputs.declare<cv::Mat>("model_points", "All points in the model.");
  }

  void configure(const tendrils& params, const tendrils& inputs, const tendrils& outputs)
  {
  }

  int process(const tendrils& inputs, const tendrils& outputs)
  {
    std::vector<cv::Point3d> new_points;
    inputs.get<cv::Mat>("points").copyTo(new_points);
    cv::Mat new_descriptors;
    inputs["descriptors"] >> new_descriptors;

    model_points_.insert(model_points_.end(), new_points.begin(), new_points.end());
    int r = model_descriptors_.rows;
    model_descriptors_.resize(r + new_descriptors.rows);
    cv::Mat target = model_descriptors_.rowRange(r, r + new_descriptors.rows);
    new_descriptors.copyTo(target);

    cv::Mat out_points(model_points_);
    outputs["model_points"] << out_points;
    outputs["model_descriptors"] << model_descriptors_;

    std::cout << "ModelBuilder: Model contains " << model_points_.size() << " points and " << model_descriptors_.rows << " descriptors." << std::endl;
    return ecto::OK;
  }

  std::vector<cv::Point3d> model_points_;
  cv::Mat model_descriptors_;
  ecto::spore<std::map<int, int> > new_dtp_;

};

}

ECTO_CELL(features3d, features3d::ModelBuilder, "ModelBuilder", "Builds up a 3d feature model.");


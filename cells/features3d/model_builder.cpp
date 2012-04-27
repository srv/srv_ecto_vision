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

    inputs.declare(&ModelBuilder::new_points_, "points", "Input source points.").required(true);
    inputs.declare(&ModelBuilder::new_descriptors_, "descriptors", "Input descriptors.").required(true);
    inputs.declare(&ModelBuilder::new_dtp_, "map", "Descriptors to points map.");
    outputs.declare(&ModelBuilder::model_points_, "model_points", "All points in the model.");
  }

  void configure(const tendrils& params, const tendrils& inputs, const tendrils& outputs)
  {
  }

  int process(const tendrils& inputs, const tendrils& outputs)
  {

    assert(source_point_indices_.size() == target_point_indices_.size());
    std::vector<cv::Point3d> source_points, target_points;
    if (source_point_indices_->size() == 0)
    {
      source_points = *source_points_;
      target_points = *target_points_;
    else
    {
      for (size_t i = 0; i < source_point_indices_->size(); ++i)
      {
        source_points.push_back(source_points_->at(source_point_indices_->at(i)));
        target_points.push_back(target_points_->at(source_point_indices_->at(i)));
      }
    }

    cv::Mat transform;
    cv::estimateAffine3D(source_points, target_points, transform, outliers, ransac_threshold_, confidence_);
    outputs["transformation"] << transform;
    return ecto::OK;
  }

  ecto::spore<std::vector<cv::Point3d> > source_points_;
  ecto::spore<std::vector<cv::Point3d> > target_points_;
  ecto::spore<std::vector<int> > source_point_indices_;
  ecto::spore<std::vector<int> > target_point_indices_;
  double ransac_threshold_;
  double confidence_;

};

}

ECTO_CELL(features3d, features3d::ModelBuilder, "ModelBuilder", "Builds up a 3d feature model.");


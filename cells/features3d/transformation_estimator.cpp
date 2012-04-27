#include <ecto/ecto.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>

namespace features3d
{

using ecto::tendrils;

struct TransformationEstimator
{
  static void declare_params(tendrils& params)
  {
    params.declare<double>("ransac_threshold", "The maximum reprojection error in RANSAC algorithm to consider a point an inlier.", 3.0);
    params.declare<double>("confidence", "The confidence level, between 0 and 1, with which the matrix is estimated.", 0.99);
  }

  static void declare_io(const tendrils& params, tendrils& inputs, tendrils& outputs)
  {
    inputs.declare(&TransformationEstimator::source_points_, "source_points", "Input source points.").required(true);
    inputs.declare(&TransformationEstimator::target_points_, "target_points", "Input target points.").required(true);
    inputs.declare(&TransformationEstimator::source_point_indices_, "source_point_indices", "Indices of input points to use.");
    inputs.declare(&TransformationEstimator::target_point_indices_, "target_point_indices", "Indices of target points to use.");
    outputs.declare<cv::Mat>("transformation", "Transformation between source and target points.");
  }

  void configure(const tendrils& params, const tendrils& inputs, const tendrils& outputs)
  {
    ransac_threshold_ = params.get<double>("ransac_threshold");
    confidence_ = params.get<double>("confidence");
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

ECTO_CELL(features3d, features3d::TransformationEstimator, "TransformationEstimator", "Estimates a transform.");


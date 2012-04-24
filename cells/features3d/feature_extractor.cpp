#include <ecto/ecto.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>

#include <feature_extraction/key_point_detector_factory.h>
#include <feature_extraction/descriptor_extractor_factory.h>

namespace features3d
{

using ecto::tendrils;

struct FeatureExtractor
{
  static void declare_params(tendrils& params)
  {
    params.declare<std::string>("detector", "The key point detector type", "SmartSURF");
    params.declare<std::string>("extractor", "The descriptor extractor type", "SmartSURF");
  }

  static void declare_io(const tendrils& params, tendrils& inputs, tendrils& outputs)
  {
    inputs.declare<cv::Mat>("image", "The input image.");
    outputs.declare<std::vector<cv::KeyPoint> >("key_points", "The detected key points.");
    outputs.declare<cv::Mat>("descriptors", "The descriptors as matrix, one row per key point");
  }

  void configure(const tendrils& params, const tendrils& inputs, const tendrils& outputs)
  {
    key_point_detector_ =
      feature_extraction::KeyPointDetectorFactory::create(params.get<std::string>("detector"));
    descriptor_extractor_ =
      feature_extraction::DescriptorExtractorFactory::create(params.get<std::string>("extractor"));
  }

  int process(const tendrils& input, const tendrils& output)
  {
    cv::Mat in;
    input["image"] >> in;
    std::vector<cv::KeyPoint> key_points;
    key_point_detector_->detect(in, key_points);
    cv::Mat descriptors;
    descriptor_extractor_->extract(in, key_points, descriptors);

    output["key_points"] << key_points;
    output["descriptors"] << descriptors;
    std::cout << key_points.size() << " key points found." << std::endl;
    return ecto::OK;
  }

  feature_extraction::KeyPointDetector::Ptr key_point_detector_;
  feature_extraction::DescriptorExtractor::Ptr descriptor_extractor_;

};

}

ECTO_CELL(features3d, features3d::FeatureExtractor, "FeatureExtractor", "Extracts key points and descriptors.");


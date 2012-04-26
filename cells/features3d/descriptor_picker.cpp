#include <ecto/ecto.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>

namespace features3d
{

using ecto::tendrils;

struct DescriptorPicker
{
  static void declare_params(tendrils& params)
  {
  }

  static void declare_io(const tendrils& params, tendrils& inputs, tendrils& outputs)
  {
    inputs.declare<cv::Mat>("descriptors", "Descriptors for training. These will be found.").required(true);
    inputs.declare<std::vector<int> >("indices", "Indices to extract.");
    outputs.declare<cv::Mat>("descriptors", "Picked descriptors.");
  }

  void configure(const tendrils& params, const tendrils& inputs, const tendrils& outputs)
  {
  }
 
  int process(const tendrils& inputs, const tendrils& outputs)
  {
    cv::Mat descriptors;
    inputs["descriptors"] >> descriptors;
    std::vector<int> indices;
    inputs["indices"] >> indices;
    cv::Mat out_descriptors(indices.size(), descriptors.cols, descriptors.type());
    for (size_t i = 0; i < indices.size(); ++i)
    {
      cv::Mat source = descriptors.row(indices[i]);
      cv::Mat destination = out_descriptors.row(i);
      source.copyTo(destination);
    }

    outputs["descriptors"] << descriptors;
    return ecto::OK;
  }

};

}

ECTO_CELL(features3d, features3d::DescriptorPicker, "DescriptorPicker", "Picks descriptors from an input matrix using input indices and constructs a new output matrix containing the picked descriptors.");


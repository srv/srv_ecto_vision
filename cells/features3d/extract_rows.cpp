#include <ecto/ecto.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>

namespace features3d
{

using ecto::tendrils;

struct ExtractRows
{
  static void declare_params(tendrils& params)
  {
  }

  static void declare_io(const tendrils& params, tendrils& inputs, tendrils& outputs)
  {
    inputs.declare<cv::Mat>("mat", "Input matrix.").required(true);
    inputs.declare(&ExtractRows::indices_, "indices", "Indices to extract.");
    outputs.declare<cv::Mat>("mat", "Output matrix, contains those rows referenced by indices only.");
  }

  void configure(const tendrils& params, const tendrils& inputs, const tendrils& outputs)
  {
  }
 
  int process(const tendrils& inputs, const tendrils& outputs)
  {
    cv::Mat input, output;
    inputs["mat"] >> input;
    if (indices_->size() == 0)
      output = input;
    else
    {
      output.create(indices_->size(), input.cols, input.type());
      for (size_t i = 0; i < indices_->size(); ++i)
      {
        cv::Mat source = input.row(indices_->at(i));
        cv::Mat destination = output.row(i);
        source.copyTo(destination);
      }
    }
    outputs["mat"] << output;
    return ecto::OK;
  }

  ecto::spore<std::vector<int> > indices_;

};

}

ECTO_CELL(features3d, features3d::ExtractRows, "ExtractRows", "Extracts rows given by indices from a matrix into a new matrix.");


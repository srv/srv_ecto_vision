#include <ecto/ecto.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>

namespace features3d
{

using ecto::tendrils;

struct MatchesToIndices
{
  static void declare_params(tendrils& params)
  {
  }

  static void declare_io(const tendrils& params, tendrils& inputs, tendrils& outputs)
  {
    inputs.declare<std::vector<cv::DMatch> >("matches", "The input matches.");
    outputs.declare<std::vector<int> >("query_indices", "The matched query indices.");
    outputs.declare<std::vector<int> >("train_indices", "The matched train indices.");
  }

  void configure(const tendrils& params, const tendrils& inputs, const tendrils& outputs)
  {
  }
 
  int process(const tendrils& input, const tendrils& output)
  {
    std::vector<cv::DMatch> matches;
    input["matches"] >> matches;
    std::vector<int> query_indices(matches.size());
    std::vector<int> train_indices(matches.size());
    for (size_t i = 0; i < matches.size(); i++ )
    {
      query_indices[i] = matches[i].queryIdx;
      train_indices[i] = matches[i].trainIdx;
    }

    output["query_indices"] << query_indices;
    output["train_indices"] << train_indices;

    return ecto::OK;
  }
};

}

ECTO_CELL(features3d, features3d::MatchesToIndices, "MatchesToIndices", "Extracts the indices from a set of matches.");


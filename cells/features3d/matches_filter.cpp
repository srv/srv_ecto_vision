#include <ecto/ecto.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>

namespace features3d
{

using ecto::tendrils;

struct MatchesFilter
{
  static void declare_params(tendrils& params)
  {
    params.declare<double>("threshold", "The threshold for threshold matching.", 0.8);
    params.declare<bool>("verbose", "Print some info during process.", false);
  }

  static void declare_io(const tendrils& params, tendrils& inputs, tendrils& outputs)
  {
    inputs.declare<cv::Mat>("match_mask", "A mask that marks allowed matches. rows = train, cols = query");
    inputs.declare<std::vector<std::vector<cv::DMatch> > >("knn_matches", "The matches to be filtered.");
    outputs.declare<std::vector<cv::DMatch> >("matches", "The filtered matches.");
  }

  void configure(const tendrils& params, const tendrils& inputs, const tendrils& outputs)
  {
    threshold_ = params.get<double>("threshold");
    verbose_ = params.get<bool>("verbose");
  }
 
  int process(const tendrils& input, const tendrils& output)
  {
    cv::Mat match_mask;
    input["match_mask"] >> match_mask;
    std::vector<std::vector<cv::DMatch> > knn_matches;
    input["knn_matches"] >> knn_matches;

    std::vector<cv::DMatch> matches;
    for (size_t m = 0; m < knn_matches.size(); m++ )
    {
      if (knn_matches[m].size() < 2) continue;
      bool match_allowed = match_mask.at<unsigned char>(
          knn_matches[m][0].trainIdx, knn_matches[m][0].queryIdx) > 0;
      if (match_allowed)
      {
        float dist1 = knn_matches[m][0].distance;
        float dist2 = knn_matches[m][1].distance;
        if (dist1 / dist2 < threshold_)
        {
          matches.push_back(knn_matches[m][0]);
        }
      }
    }

    if (verbose_)
    {
      std::cout << knn_matches.size() << " matches before, " << matches.size() << " matches after filtering." << std::endl;
    }

    output["matches"] << matches;


    return ecto::OK;
  }

  double threshold_;
  bool verbose_;

};

}

ECTO_CELL(features3d, features3d::MatchesFilter, "MatchesFilter", "Filters knn matches by threshold and match mask, output are simple matches.");


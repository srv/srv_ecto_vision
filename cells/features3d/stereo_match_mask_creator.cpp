#include <ecto/ecto.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <numeric>

#include <feature_extraction/key_point_detector_factory.h>
#include <feature_extraction/descriptor_extractor_factory.h>

namespace features3d
{

using ecto::tendrils;

class Stats
{
  public:

    void add(double val)
    {
      vals_.push_back(val);
    }

    int num() const { 
      return vals_.size();
    }

    double min() const 
    { 
      return *std::min_element(vals_.begin(), vals_.end());
    }

    double max() const
    {
      return *std::max_element(vals_.begin(), vals_.end());
    }

    double mean() const
    {
      return std::accumulate(vals_.begin(), vals_.end(), 0.0) / vals_.size();
    }

  private:

    std::vector<double> vals_;
};

std::ostream& operator<<(std::ostream& ostr, const Stats& stats)
{
  ostr << "num: " << stats.num() << ", min: " << stats.min() << ", max: " << stats.max() << ", mean: " << stats.mean();
  return ostr;
}


struct StereoMatchMaskCreator
{
  static void declare_params(tendrils& params)
  {
    params.declare<double>("max_y_diff", "The maximum pixel difference in y between matching key points", 2.0);
    params.declare<double>("max_angle_diff", "The maximum difference in angle between matching key points (Rad)", 5.0/180.0*M_PI);
    params.declare<double>("max_size_ratio", "The maximum allowed ratio between sizes of matching key points, must be > 1", 1.5);
  }

  static void declare_io(const tendrils& params, tendrils& inputs, tendrils& outputs)
  {
    inputs.declare(&StereoMatchMaskCreator::key_points_left_, "key_points_left", "The key points of the left image");
    inputs.declare(&StereoMatchMaskCreator::key_points_right_, "key_points_right", "The key points of the right image");
    outputs.declare<cv::Mat>("match_mask", "The computed match mask");
  }

  void configure(const tendrils& params, const tendrils& inputs, const tendrils& outputs)
  {
    max_y_diff_ = params.get<double>("max_y_diff");
    max_angle_diff_ = params.get<double>("max_angle_diff") / M_PI * 180.0;
    max_size_ratio_ = params.get<double>("max_size_ratio");
    if (max_size_ratio_ < 1.0) throw std::runtime_error("invalid max_size_ratio, must be > 1.0");
  }

  int process(const tendrils& input, const tendrils& output)
  {
    cv::Mat match_mask;
    match_mask.create(key_points_left_->size(), key_points_right_->size(), CV_8UC1);
    Stats y_diff_stats, angle_diff_stats, disparity_stats, size_ratio_stats;
    for (int r = 0; r < match_mask.rows; ++r)
    {
      const cv::KeyPoint& keypoint_left = key_points_left_->at(r);
      for (int c = 0; c < match_mask.cols; ++c)
      {
        const cv::KeyPoint& keypoint_right = key_points_right_->at(c);
        bool allow_match = false;
        // y_diff check, filters out most mismatches
        double y_diff = fabs(keypoint_left.pt.y - keypoint_right.pt.y);
        y_diff_stats.add(y_diff);
        if (y_diff <= max_y_diff_)
        {
          // angle check
          // NOTE: cv::KeyPoint carries angle information in degrees
          double angle_diff = std::abs(keypoint_left.angle - keypoint_right.angle);
          angle_diff = std::min(360 - angle_diff, angle_diff);
          angle_diff_stats.add(angle_diff);
          if (angle_diff <= max_angle_diff_)
          {
            // disparity check
            double disparity = keypoint_left.pt.x - keypoint_right.pt.x;
            disparity_stats.add(disparity);
            if (disparity >= 0)
            {
              // size check
              double ratio = keypoint_left.size > keypoint_right.size ?
                keypoint_left.size / keypoint_right.size :
                keypoint_right.size / keypoint_left.size;
              size_ratio_stats.add(ratio);
              if (ratio <= max_size_ratio_)
              {
                allow_match = true;
              }
            }
          }
        }

        if (allow_match)
        {
          match_mask.at<unsigned char>(r, c) = 255;
        }
        else
        {
          match_mask.at<unsigned char>(r, c) = 0;
        }
      }
    } 
    output["match_mask"] << match_mask;
    /*
    std::cout << "y diff stats: " << y_diff_stats << std::endl;
    std::cout << "angle diff stats: " << angle_diff_stats << std::endl;
    std::cout << "disparity diff stats: " << disparity_stats << std::endl;
    std::cout << "size ratio stats: " << size_ratio_stats << std::endl;
    */

    return ecto::OK;
  }

  ecto::spore<std::vector<cv::KeyPoint> > key_points_left_, key_points_right_;
  double max_y_diff_;
  double max_angle_diff_;
  double max_size_ratio_;

};

}

ECTO_CELL(features3d, features3d::StereoMatchMaskCreator, "StereoMatchMaskCreator", "Creates a match mask for valid stereo key point pairs.");


#include <ecto/ecto.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>

namespace features3d
{

using ecto::tendrils;

struct CorrespondenceFinder
{
  static void declare_params(tendrils& params)
  {
    params.declare<double>("matching_threshold", "Distance ratio threshold to accept descriptor matches", 0.8);
  }

  static void declare_io(const tendrils& params, tendrils& inputs, tendrils& outputs)
  {
    inputs.declare(&CorrespondenceFinder::knn_matches_, "knn_matches", "Input knn matches between descriptors.").required(true);
    inputs.declare(&CorrespondenceFinder::dtp_train_, "train_index_map", "Descriptors to (point) indices map for training set.");
    inputs.declare(&CorrespondenceFinder::dtp_query_, "query_index_map", "Descriptors to (point) indices map for query set.");
    outputs.declare(&CorrespondenceFinder::train_point_indices_, "train_point_indices", "Correspondence indices of training points.");
    outputs.declare(&CorrespondenceFinder::query_point_indices_, "query_point_indices", "Correspondence indices of query points.");
  }

  void configure(const tendrils& params, const tendrils& inputs, const tendrils& outputs)
  {
    matching_threshold_ = params.get<double>("matching_threshold");
  }

  int remap(int index, const std::map<int, int>& map)
  {
    if(map.empty()) return index;
    std::map<int, int>::const_iterator iter = map.find(index);
    assert(iter != map.end());
    return iter->second;
  }
 
  int process(const tendrils& inputs, const tendrils& outputs)
  {
    train_point_indices_->clear();
    query_point_indices_->clear();
    // train = all model points and descriptors
    // query = current 3d features
    for (size_t i = 0; i < knn_matches_->size(); ++i)
    {
      const std::vector<cv::DMatch>& knn_match = knn_matches_->at(i);
      if (knn_match.size() == 0) continue;
      bool correspondence_found = false;
      if (knn_match.size() == 1)
        correspondence_found = true;
      else if (knn_match.size() == 2) // the normal case
      {
        if (knn_match[0].trainIdx == knn_match[1].trainIdx)
          correspondence_found = true;
        else if (knn_match[0].distance / knn_match[1].distance < matching_threshold_)
          correspondence_found = true;
      }
      if (correspondence_found)
      {
        train_point_indices_->push_back(remap(knn_match[0].trainIdx, *dtp_train_));
        query_point_indices_->push_back(remap(knn_match[0].queryIdx, *dtp_query_));
      }
    }
    return ecto::OK;
  }

  ecto::spore<std::vector<std::vector<cv::DMatch> > > knn_matches_;
  ecto::spore<std::map<int, int> > dtp_train_;
  ecto::spore<std::map<int, int> > dtp_query_;
  ecto::spore<std::vector<int> > train_point_indices_;
  ecto::spore<std::vector<int> > query_point_indices_;
  double matching_threshold_;

};

}

ECTO_CELL(features3d, features3d::CorrespondenceFinder, "CorrespondenceFinder", "Finds correspondences between matched descriptors, allows remapping of indices to allow multiple descriptors per point.");


#include <ecto_pcl/ecto_pcl.hpp>
#include <ecto_pcl/pcl_cell_dual_inputs.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#include <pcl/kdtree/kdtree_flann.h>
#pragma GCC diagnostic pop

namespace features3d
{

using ecto::tendrils;

struct ModelBuilder
{
  static const std::string SecondInputName;
  static const std::string SecondInputDescription;

  static void declare_params(tendrils& params)
  {
  }

  static void declare_io(const tendrils& params, tendrils& inputs, tendrils& outputs)
  {
    inputs.declare<cv::Mat>("new_descriptors", "Input descriptors.").required(true);
    inputs.declare<cv::Mat>("model_descriptors", "Input descriptors.");
    inputs.declare(&ModelBuilder::dtp_model_in_, "index_map", "Descriptors to point indices map of the model");
    outputs.declare(&ModelBuilder::dtp_model_out_, "index_map", "Descriptors to point indices map of the augmented model.");
    outputs.declare<cv::Mat>("model_descriptors", "Augmented model descriptors");
    outputs.declare<ecto::pcl::PointCloud>("model_points", "All points in the model.");
  }

  void configure(const tendrils& params, const tendrils& inputs, const tendrils& outputs)
  {
    output_ = outputs["model_points"];
  }

  template <typename PointT>
  int process(const tendrils& inputs, const tendrils& outputs, boost::shared_ptr<const pcl::PointCloud<PointT> >& new_points,
      boost::shared_ptr<const pcl::PointCloud<PointT> >& model_points)
  {
    assert(new_points);
    cv::Mat new_descriptors;
    inputs["new_descriptors"] >> new_descriptors;
    cv::Mat model_descriptors;
    inputs["model_descriptors"] >> model_descriptors;

    pcl::PointCloud<PointT> joined_points;
    if (model_points)
    {
      joined_points = *model_points;

      *dtp_model_out_ = *dtp_model_in_;
      pcl::KdTreeFLANN<PointT> tree;
      tree.setInputCloud(model_points);
      for (size_t i = 0; i < new_points->size(); ++i)
      {
        std::vector<int> k_indices(1);
        std::vector<float> k_distances(1);
        int found_k = tree.nearestKSearch(new_points->at(i), 1, k_indices, k_distances);
        if (found_k == 1 && k_distances[0] < 0.01)
        {
            // same point, add only descriptor if different
        }
        else
        {
          // new point, add both descriptor and point
          joined_points.push_back(new_points->at(i));
          size_t point_index = joined_points.size() - 1;
          model_descriptors.resize(model_descriptors.rows + 1);
          cv::Mat target = model_descriptors.row(model_descriptors.rows - 1);
          cv::Mat source = new_descriptors.row(i);
          source.copyTo(target);
          (*dtp_model_out_)[model_descriptors.rows - 1] = point_index;
        }
      }
    }
    else
    {
      // model empty, initialize with first incoming
      joined_points = *new_points;
      model_descriptors = new_descriptors;
      for (size_t i = 0; i < joined_points.size(); ++i)
        (*dtp_model_out_)[i] = i;
    }

    *output_ = ecto::pcl::xyz_cloud_variant_t(joined_points.makeShared());
    outputs["model_descriptors"] << model_descriptors;

    std::cout << "ModelBuilder: Model contains " << model_points->size() << " points and " << model_descriptors.rows << " descriptors." << std::endl;
    return ecto::OK;
  }

  ecto::spore<ecto::pcl::PointCloud> output_;
  ecto::spore<std::map<int, int> > dtp_model_in_;
  ecto::spore<std::map<int, int> > dtp_model_out_;

};

const std::string ModelBuilder::SecondInputName = "model_points";
const std::string ModelBuilder::SecondInputDescription = "The collected model points";


}

ECTO_CELL(features3d, ecto::pcl::PclCellDualInputs<features3d::ModelBuilder>, "ModelBuilder", "Builds up a 3d feature model.");


#!/usr/bin/python
import ecto #ecto core library
from ecto.opts import scheduler_options, run_plasm, cell_options
from srv_ecto_vision import features3d
import ecto_opencv
from ecto_opencv.highgui import imshow, FPSDrawer, ImageReader
from ecto_opencv.features2d import DrawKeypoints, DrawMatches, KeypointsToMat
from ecto_pcl import CloudViewer
import argparse

debug = True

def do_ecto():
  parser = argparse.ArgumentParser(description="Stereo Feature Extractor")

  # add cell options
  feature_extractor_factory = cell_options(parser, features3d.FeatureExtractor, prefix="fe")

  # add scheduler options
  group = parser.add_argument_group("ecto scheduler options")
  scheduler_options(group)

  options = parser.parse_args()

  img_dir = "/home/stwirth/uib-ros/bagfiles/fugu-c/images/"
  reader_left = ImageReader(path=img_dir, match=".*left.*\.png")
  reader_right = ImageReader(path=img_dir, match=".*right.*\.png")

  # setup cells
  feature_extractor_left  = feature_extractor_factory(options)
  feature_extractor_right = feature_extractor_factory(options)

  match_mask_creator = features3d.StereoMatchMaskCreator()
  descriptor_matcher = features3d.DescriptorMatcherKnn()
  matches_filter = features3d.MatchesFilter(verbose=True)

  matches_drawer = DrawMatches()
  kp2mat_left = KeypointsToMat()
  kp2mat_right = KeypointsToMat()

  matches_to_indices = features3d.MatchesToIndices()
  key_points_to_points_left = features3d.KeyPointsToPoints()
  key_points_to_points_right = features3d.KeyPointsToPoints()

  stereo_depth_estimator = features3d.StereoDepthEstimator(
      camera_info_left_file="/home/stwirth/uib-ros/fugu/fugu_configurations/camera_comp_wide/calibration_cylinder_water_left.yaml",
      camera_info_right_file="/home/stwirth/uib-ros/fugu/fugu_configurations/camera_comp_wide/calibration_cylinder_water_right.yaml");
  points_to_point_cloud = features3d.PointsToPointCloud()

  descriptor_picker = features3d.ExtractRows()

  model_builder = features3d.ModelBuilder()

  model_points_source, model_points_sink = ecto.EntangledPair(value=model_builder.inputs.at('model_points'))
  model_descriptors_source, model_descriptors_sink = ecto.EntangledPair(value=model_builder.inputs.at('model_descriptors'))

  #setup the processing graph
  graph = [

      reader_left["image"] >> feature_extractor_left["image"],
      reader_right["image"] >> feature_extractor_right["image"],

      # feature extraction & matching
      feature_extractor_left["key_points"] >> match_mask_creator["key_points_left"],
      feature_extractor_right["key_points"] >> match_mask_creator["key_points_right"],
      feature_extractor_left["descriptors"] >> descriptor_matcher["train_descriptors"],
      feature_extractor_right["descriptors"] >> descriptor_matcher["test_descriptors"],
      descriptor_matcher["knn_matches"] >> matches_filter["knn_matches"],
      match_mask_creator["match_mask"] >> matches_filter["match_mask"],

      # display matches
      matches_filter["matches"] >> matches_drawer["matches"],
      feature_extractor_left["key_points"] >> kp2mat_left["keypoints"],
      feature_extractor_right["key_points"] >> kp2mat_right["keypoints"],
      kp2mat_left["points"] >> matches_drawer["train"],
      kp2mat_right["points"] >> matches_drawer["test"],
      reader_left["image"] >> matches_drawer["train_image"],
      reader_right["image"] >> matches_drawer["test_image"],
      matches_drawer["output"] >> imshow(name="matches")["image"],

      # match_mask_creator["match_mask"] >> imshow(name="match mask")["image"],
      matches_filter["matches"] >> matches_to_indices["matches"],

      matches_to_indices["train_indices"] >> key_points_to_points_left["indices"],
      matches_to_indices["query_indices"] >> key_points_to_points_right["indices"],
      feature_extractor_left["key_points"] >> key_points_to_points_left["key_points"],
      feature_extractor_right["key_points"] >> key_points_to_points_right["key_points"],

      key_points_to_points_left["points"] >> stereo_depth_estimator["points_left"],
      key_points_to_points_right["points"] >> stereo_depth_estimator["points_right"],
      
      # point cloud display
      stereo_depth_estimator["points_3d"] >> points_to_point_cloud["points_3d"],
      reader_left["image"] >> points_to_point_cloud["image"],
      key_points_to_points_left["points"] >> points_to_point_cloud["image_points"],
      points_to_point_cloud["point_cloud"] >> CloudViewer(window_name="point cloud")["input"],

      matches_to_indices["train_indices"] >> descriptor_picker["indices"],
      feature_extractor_left["descriptors"] >> descriptor_picker["mat"],
      descriptor_picker["mat"] >> imshow(name="picked descriptors")["image"],

      points_to_point_cloud["point_cloud"] >> model_builder["input"],
      descriptor_picker["mat"] >> model_builder["new_descriptors"],

      # feedback loop
      model_points_source["out"] >> model_builder["model_points"],
      model_builder["model_points"] >> model_points_sink["in"],

      model_descriptors_source["out"] >> model_builder["model_descriptors"],
      model_builder["model_descriptors"] >> model_descriptors_sink["in"],

      ]
              
  #instantiate a plasm
  plasm = ecto.Plasm()
  plasm.connect(graph)

  sched = ecto.schedulers.Singlethreaded(plasm)

  run_plasm(options, plasm, locals=vars())
  
if __name__ == '__main__':
  import sys
  do_ecto()


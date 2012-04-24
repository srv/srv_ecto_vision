#!/usr/bin/python
import ecto #ecto core library
from ecto.opts import scheduler_options, run_plasm, cell_options
from srv_ecto_vision import features3d
import ecto_ros
from ecto_ros import ecto_sensor_msgs
import ecto_opencv
from ecto_opencv.highgui import imshow, FPSDrawer
from ecto_opencv.features2d import DrawKeypoints
import argparse

#short names
ImageSub = ecto_sensor_msgs.Subscriber_Image
InfoSub = ecto_sensor_msgs.Subscriber_CameraInfo

debug = True

def do_ecto():
  parser = argparse.ArgumentParser(description="Stereo Feature Extractor")

  # add cell options
  feature_extractor_factory = cell_options(parser, features3d.FeatureExtractor, prefix="fe")

  # add scheduler options
  group = parser.add_argument_group("ecto scheduler options")
  scheduler_options(group)

  options = parser.parse_args()

  # setup ros subscribers
  subscriptions = dict(image_left  = ImageSub(topic_name="image_left",  queue_size=0),
                       #info_left   = InfoSub (topic_name="camera_info_left", queue_size=0),
                       image_right = ImageSub(topic_name="image_right", queue_size=0),
                       #info_right  = InfoSub (topic_name="camera_info_right", queue_size=0)
                       )
  sync = ecto_ros.Synchronizer("Synchronizer", subs=subscriptions)

  # setup cells
  image2mat_left = ecto_ros.Image2Mat()
  image2mat_right = ecto_ros.Image2Mat()
  feature_extractor_left  = feature_extractor_factory(options)
  feature_extractor_right = feature_extractor_factory(options)

  match_mask_creator = features3d.StereoMatchMaskCreator()
  descriptor_matcher = features3d.DescriptorMatcherKnn()
  matches_filter = features3d.MatchesFilter(verbose=True)

  #setup the processing graph
  graph = [
      sync["image_left"] >> image2mat_left["image"],
      image2mat_left["image"] >> feature_extractor_left["image"],
      sync["image_right"] >> image2mat_right["image"],
      image2mat_right["image"] >> feature_extractor_right["image"],

      feature_extractor_left["key_points"] >> match_mask_creator["key_points_left"],
      feature_extractor_right["key_points"] >> match_mask_creator["key_points_right"],

      feature_extractor_left["descriptors"] >> descriptor_matcher["train_descriptors"],
      feature_extractor_right["descriptors"] >> descriptor_matcher["test_descriptors"],

      descriptor_matcher["knn_matches"] >> matches_filter["knn_matches"],
      match_mask_creator["match_mask"] >> matches_filter["match_mask"],

      match_mask_creator["match_mask"] >> imshow(name="match mask")["image"],
      ]
              
  #instantiate a plasm
  plasm = ecto.Plasm()
  plasm.connect(graph)

  sched = ecto.schedulers.Singlethreaded(plasm)

  run_plasm(options, plasm, locals=vars())
  
if __name__ == '__main__':
  import sys
  # init ros (does topic remapping)
  ecto_ros.init(sys.argv, "stereo_feature_extraction")
  do_ecto()


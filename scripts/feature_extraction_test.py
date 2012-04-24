import ecto #ecto core library
from ecto.opts import scheduler_options, run_plasm, cell_options
from srv_ecto_vision import features3d
import ecto_ros
import ecto_opencv
from ecto_ros import ecto_sensor_msgs
from ecto_opencv.highgui import imshow, FPSDrawer
from ecto_opencv.features2d import DrawKeypoints
import argparse

#short names
ImageSub = ecto_sensor_msgs.Subscriber_Image
InfoSub = ecto_sensor_msgs.Subscriber_CameraInfo

debug = True

def do_ecto():
  parser = argparse.ArgumentParser(description="Feature Extractor")

  # add cell options
  feature_extractor_factory = cell_options(parser, features3d.FeatureExtractor, prefix="fe")

  # add scheduler options
  group = parser.add_argument_group("ecto scheduler options")
  scheduler_options(group)

  options = parser.parse_args()

  # setup ros subscribers
  subscriptions = dict(image=ImageSub(topic_name="image",queue_size=0),
                       info=InfoSub(topic_name="camera_info",queue_size=0))
  sync = ecto_ros.Synchronizer("Synchronizer", subs=subscriptions)

  image_converter = ecto_ros.Image2Mat()
  mat2img = ecto_ros.Mat2Image()
  drawer = DrawKeypoints()
  feature_extractor = feature_extractor_factory(options)
  fps = FPSDrawer()
  image_pub = ecto_sensor_msgs.Publisher_Image(topic_name="image_with_key_points")

  #setup the processing graph
  graph = [
      sync["image"] >> image_converter["image"],
      image_converter["image"] >> feature_extractor["image"],
      feature_extractor["key_points"] >> drawer["keypoints"],
      image_converter["image"] >> drawer["image"],
      drawer["image"] >> fps["image"],
      fps["image"] >> imshow(name="key points")["image"],
      fps["image"] >> mat2img["image"],
      mat2img["image"] >> image_pub["input"]
      ]
              
  #instantiate a plasm
  plasm = ecto.Plasm()
  plasm.connect(graph)

  #sched = ecto.schedulers.Singlethreaded(plasm)
  #sched.execute(niter=10)
  #print sched.stats()

  run_plasm(options, plasm, locals=vars())
  
if __name__ == '__main__':
  import sys
  # init ros (does topic remapping)
  ecto_ros.init(sys.argv, "hello")
  do_ecto()


# TODO remove the following after electric
if (ROS_ELECTRIC_FOUND)
    find_ros_package(feature_extraction)
else()
    find_package(ROS QUIET COMPONENTS feature_extraction)
endif()

find_package(ecto_pcl)

ectomodule(features3d
  descriptor_matcher_knn.cpp
  feature_extractor.cpp
  matches_filter.cpp
  module.cpp
  stereo_match_mask_creator.cpp
  matches_to_indices.cpp
  key_points_to_points.cpp
  stereo_depth_estimator.cpp
  points_to_point_cloud.cpp
  sync_check.cpp
  correspondence_finder.cpp
  extract_rows.cpp
  transform.cpp
  model_builder.cpp
)

include_directories(SYSTEM ${ROS_INCLUDE_DIRS} ${ecto_pcl_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)
link_ecto(features3d
    ${OpenCV_LIBS}
    ${feature_extraction_LIBRARIES}
    )

install_ecto_module(features3d)


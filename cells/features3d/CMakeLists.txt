# TODO remove the following after electric
if (ROS_ELECTRIC_FOUND)
    find_ros_package(feature_extraction)
else()
    find_package(ROS QUIET COMPONENTS feature_extraction)
endif()

ectomodule(features3d
  descriptor_matcher_knn.cpp
  feature_extractor.cpp
  matches_filter.cpp
  module.cpp
  stereo_match_mask_creator.cpp
)

include_directories(SYSTEM ${ROS_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)
link_ecto(features3d
    ${OpenCV_LIBS}
    ${feature_extraction_LIBRARIES}
    )

install_ecto_module(features3d)

